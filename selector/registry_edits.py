"""Recognize inert edits whose import-time effects are registry updates."""

import ast
from difflib import SequenceMatcher


def added_expressions(old, new, known):
    """Return fresh Expression/Func declarations with literal class bodies."""
    imports = {alias.name for node in old.body if isinstance(node, ast.ImportFrom)
               and node.module == "sqlglot.expressions.core" and not node.level
               for alias in node.names if alias.asname is None}
    if "Expression" not in imports:
        return {}
    additions = {}
    before_names = {node.name for node in old.body if isinstance(node, ast.ClassDef)}
    retained = []
    for node in new.body:
        if not isinstance(node, ast.ClassDef) or node.name in before_names:
            retained.append(node)
            continue
        bases = tuple(base.id for base in node.bases if isinstance(base, ast.Name))
        if (len(bases) != len(node.bases) or bases not in (("Expression",), ("Expression", "Func"))
                or not set(bases) <= imports or node.keywords or node.decorator_list
                or getattr(node, "type_params", ()) or node.name.startswith("_")
                or node.name.lower() in known or node.name.lower() in additions):
            return {}
        names = set()
        for i, stmt in enumerate(node.body):
            if isinstance(stmt, ast.Pass) or (i == 0 and ast.get_docstring(node) is not None):
                continue
            if (not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1
                    or not isinstance(stmt.targets[0], ast.Name)):
                return {}
            name, value = stmt.targets[0].id, stmt.value
            if name in names or not _expression_field(name, value):
                return {}
            names.add(name)
        additions[node.name.lower()] = (node.name, "Func" in bases)
    original = ast.Module(body=retained, type_ignores=new.type_ignores)
    return additions if ast.dump(original) == ast.dump(old) else {}


def _expression_field(name, value):
    if name == "arg_types" and isinstance(value, ast.Dict):
        return (all(isinstance(key, ast.Constant) and type(key.value) is str for key in value.keys)
                and all(isinstance(flag, ast.Constant) and type(flag.value) is bool for flag in value.values))
    if name == "_sql_names" and isinstance(value, (ast.Tuple, ast.List)):
        return all(isinstance(item, ast.Constant) and type(item.value) is str for item in value.elts)
    return name == "is_var_len_args" and isinstance(value, ast.Constant) and type(value.value) is bool


def _dtype_name(node):
    if not (isinstance(node, ast.Dict) and len(node.keys) == 1
            and isinstance(node.keys[0], ast.Constant) and node.keys[0].value == "returns"):
        return None
    enum = node.values[0]
    if not (isinstance(enum, ast.Attribute) and isinstance(enum.value, ast.Attribute)
            and enum.value.attr == "DType" and isinstance(enum.value.value, ast.Name)
            and enum.value.value.id == "exp"):
        return None
    return enum.attr


def _exp_name(node):
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "exp":
        return node.attr
    return None


def _metadata_entry(key, value):
    if key is not None:
        name = _exp_name(key)
        dtype = _dtype_name(value)
        if name is None or dtype is None:
            return None
        return dtype, {name}, ("direct", dtype, name)
    if not isinstance(value, ast.DictComp) or len(value.generators) != 1:
        return None
    loop = value.generators[0]
    if not (isinstance(value.key, ast.Name) and isinstance(loop.target, ast.Name)
            and loop.target.id == value.key.id and not loop.ifs and not loop.is_async
            and isinstance(loop.iter, ast.Set)):
        return None
    names = {_exp_name(node) for node in loop.iter.elts}
    if not names or None in names:
        return None
    dtype = _dtype_name(value.value)
    if dtype is not None:
        return dtype, names, ("returns", dtype)
    if isinstance(value.value, ast.Dict):
        return None, names, ("comp", ast.dump(value.value))
    return None


def changed_metadata(old, new):
    """Isolate expression memberships of constant return-type table entries."""
    def assignment(tree):
        found = []
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "EXPRESSION_METADATA"):
                found.append(node)
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "EXPRESSION_METADATA":
                found.append(node)
        return found[0].value if len(found) == 1 and isinstance(found[0].value, ast.Dict) else None

    left, right = assignment(old), assignment(new)
    if left is None or right is None:
        return set(), set()

    def entries(mapping):
        return {i: entry for i, (key, value) in enumerate(zip(mapping.keys, mapping.values))
                if (entry := _metadata_entry(key, value)) is not None}

    before, after = entries(left), entries(right)
    # A group's order among the groups can determine shadowing. Compare
    # signatures after removing only groups with no keys in the other version.
    def labels(mapping, groups):
        return [groups[i][2] if i in groups else (ast.dump(key) if key else None, ast.dump(value))
                for i, (key, value) in enumerate(zip(mapping.keys, mapping.values))]

    expressions, dtypes = set(), set()
    removed, added = set(), set()
    for op, i, j, k, l in SequenceMatcher(None, labels(left, before), labels(right, after), autojunk=False).get_opcodes():
        if op == "equal":
            for a, b in zip(range(i, j), range(k, l)):
                if a in before and b in after and before[a][1] != after[b][1]:
                    expressions.update(before[a][1] ^ after[b][1])
                    if after[b][0] is not None:
                        dtypes.add(after[b][0])
                    right.values[b] = left.values[a]
            continue
        for groups, indices, dropped in ((before, range(i, j), removed), (after, range(k, l), added)):
            for idx in indices:
                if idx not in groups or groups[idx][0] is None:
                    return set(), set()
                dtype, names, _ = groups[idx]
                expressions.update(names)
                dtypes.add(dtype)
                dropped.add(idx)
    for mapping, dropped in ((left, removed), (right, added)):
        mapping.keys = [key for i, key in enumerate(mapping.keys) if i not in dropped]
        mapping.values = [value for i, value in enumerate(mapping.values) if i not in dropped]
    return (expressions, dtypes) if ast.dump(old) == ast.dump(new) else (set(), set())
