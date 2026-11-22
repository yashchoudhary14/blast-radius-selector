"""Recognize pure module bindings and locate their readers in pinned query traces."""

import ast
import functools
from collections import Counter

from .source_equivalence import _binds, _same_directives_text_only


# SQLGlot's unchanged package preamble loads collections.abc, builtins and
# typing before importing its submodules. Typing itself imports functools.
CACHED_IMPORTS = {"builtins", "collections", "functools", "typing"}


def literal(node):
    if isinstance(node, ast.SetComp):
        # A comprehension over exact strings calls only the built-in str.lower.
        if len(node.generators) != 1:
            return False
        loop = node.generators[0]
        if loop.ifs or loop.is_async or not isinstance(loop.target, ast.Name):
            return False
        item = ast.parse(f"{loop.target.id}.lower()", mode="eval").body
        return (ast.dump(node.elt) == ast.dump(item) and isinstance(loop.iter, ast.Tuple)
                and all(isinstance(s, ast.Constant) and type(s.value) is str for s in loop.iter.elts))
    if any(isinstance(child, ast.Call) for child in ast.walk(node)):
        return False
    try:
        ast.literal_eval(node)
    except (ValueError, TypeError):
        return False
    return True


def declaration(node, postponed):
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
    elif isinstance(node, ast.AnnAssign) and node.simple and postponed and node.value is not None:
        target = node.target
    else:
        return None
    if isinstance(target, ast.Name) and target.id.isupper() and not target.id.startswith("__") and literal(node.value):
        return target.id
    return None


def cached_import(node):
    if isinstance(node, ast.ImportFrom) and not node.level and node.module in CACHED_IMPORTS:
        return all(not alias.name.startswith("_") and alias.name != "*" for alias in node.names)
    return isinstance(node, ast.Import) and all(alias.name in CACHED_IMPORTS for alias in node.names)


def preamble(tree):
    start = int(ast.get_docstring(tree, clean=False) is not None)
    while (start < len(tree.body) and isinstance(tree.body[start], ast.ImportFrom)
           and tree.body[start].module == "__future__"):
        start += 1
    end = start
    while end < len(tree.body) and isinstance(tree.body[end], (ast.Import, ast.ImportFrom)):
        end += 1
    return tree.body[start:end]


def new_private_functions(old, new, postponed):
    old_names = {node.name for node in old.body if isinstance(node, ast.FunctionDef)}
    functions = [node for node in new.body
                 if isinstance(node, ast.FunctionDef) and node.name not in old_names
                 and node.name.startswith("_") and not node.name.startswith("__")
                 and not node.decorator_list and not getattr(node, "type_params", ())
                 and not node.args.defaults and not any(value is not None for value in node.args.kw_defaults)
                 and (postponed or (node.returns is None and not any(
                     arg.annotation for arg in ast.walk(node.args) if isinstance(arg, ast.arg))))]
    names = {node.name for node in functions}
    return functions if len(functions) == len(names) else []


def removed_private_functions(before, after):
    """Return removed bindings whose definition does not execute user code."""
    old, new = ast.parse(before), ast.parse(after)
    postponed = any(isinstance(node, ast.ImportFrom) and node.module == "__future__"
                    and any(alias.name == "annotations" for alias in node.names) for node in old.body)
    gone = new_private_functions(new, old, postponed)
    if not gone or any(isinstance(node, ast.Name) and node.id == "__all__" for node in ast.walk(old)):
        return set()
    old.body = [node for node in old.body if node not in gone]
    if ast.dump(old) == ast.dump(new) and _same_directives_text_only(before, after):
        return {fn.name for fn in gone}
    return set()


def import_order(statements):
    """Preserve each uncached import's position relative to cached bindings."""
    sections = []
    cached = []
    for statement in statements:
        value = ast.dump(statement)
        if cached_import(statement):
            cached.append(value)
        else:
            sections.extend((sorted(cached), value))
            cached.clear()
    return [*sections, sorted(cached)]


def bindings(before, after, path):
    """Return changed values, fresh names, and the applicability of the proof."""
    old, new = ast.parse(before), ast.parse(after)
    postponed = any(isinstance(node, ast.ImportFrom) and node.module == "__future__"
                    and any(alias.name == "annotations" for alias in node.names)
                    for node in old.body)
    allow_imports = path.startswith("sqlglot/") and path != "sqlglot/__init__.py"
    old_imports = preamble(old) if allow_imports else []
    new_imports = preamble(new) if allow_imports else []
    original = {ast.dump(stmt) for stmt in old_imports}
    added = []
    for node in new_imports:
        # Only accept concrete, eagerly populated attributes of functools.
        if cached_import(node) and isinstance(node, ast.ImportFrom) and node.module == "functools" and all(
            alias.name in vars(functools) for alias in node.names
        ) and ast.dump(node) not in original:
            added.append(node)
    fresh = {alias.asname or alias.name for node in added for alias in node.names}
    retained = [node for node in new_imports if node not in added]
    if import_order(old_imports) != import_order(retained):
        return None
    imported = [alias.asname or alias.name.partition(".")[0] for node in new_imports for alias in node.names]
    if len(imported) != len(set(imported)):
        return None
    if new_imports:
        start = new.body.index(new_imports[0])
        new.body[start:start + len(new_imports)] = old_imports
    moved = list(map(ast.dump, old_imports)) != list(map(ast.dump, retained))
    functions = new_private_functions(old, new, postponed) if allow_imports else []
    fresh.update(fn.name for fn in functions)

    def declarations(tree):
        rows = [(name, node) for node in tree.body if (name := declaration(node, postponed))]
        counts = Counter(name for name, _ in rows)
        return {name: node for name, node in rows if counts[name] == 1}

    left, right = declarations(old), declarations(new)
    values = {name for name in left.keys() & right.keys()
              if ast.dump(left[name].value) != ast.dump(right[name].value)}
    additions = right.keys() - left.keys()
    fresh.update(additions)
    if not (values or fresh or moved):
        return None
    if any(_binds(node, name) for name in fresh for node in ast.walk(old)):
        return None
    if any(name.startswith("__") for name in fresh):
        return None
    for name in values:
        right[name].value = left[name].value
    omitted = {right[name] for name in additions} | set(functions)
    new.body = [node for node in new.body if node not in omitted]
    if ast.dump(old) != ast.dump(new) or not _same_directives_text_only(before, after):
        return None
    return values, fresh


def reference_lines(analyzer, path, name, preserve_binding):
    lines = set(analyzer.state_refs.refs.get(name, {}).get(path, ()))
    if preserve_binding and lines:
        tree, _ = analyzer.old_tree_and_lines(path)
        counts = Counter(node.lineno for node in tree.body)
        for statement in tree.body:
            if not isinstance(statement, ast.ImportFrom) or counts[statement.lineno] != 1:
                continue
            # Copying an existing binding is safe. A renamed copy needs its
            # own reference analysis, so its import line remains a reader.
            aliases = [alias for alias in statement.names if alias.name == name]
            if aliases and all(alias.asname is None for alias in aliases):
                lines.discard(statement.lineno)
    return lines
