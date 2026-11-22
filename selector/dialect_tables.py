"""Prove concrete table edits cannot reach a pristine dispatch trace."""

import ast
import gzip
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path, PurePosixPath

from .config_edits import added_methods
from .dialect_reads import SITES
from .source_equivalence import source_directives


TABLES = {
    "parsers": frozenset({"FUNCTIONS", "FUNCTION_PARSERS", "PROPERTY_PARSERS", "RANGE_PARSERS"}),
    "generators": frozenset({"TRANSFORMS"}),
}


def same_directive_lines(before, after):
    original, updated = source_directives(before), source_directives(after)
    return (original is not None and updated is not None
            and tuple(row[1:] for row in original) == tuple(row[1:] for row in updated))


def copied_dialect_values(old, new):
    def view(tree):
        rows = [node for node in tree.body if isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "DIALECT_MODULE_NAMES"]
        return rows[0] if len(rows) == 1 else None

    before, after = (view(tree) for tree in (old, new))
    if before is None or after is None or ast.unparse(after.value) not in {
        "dict(MODULE_BY_DIALECT).values()", "MODULE_BY_DIALECT.copy().values()",
    }:
        return False
    # The pinned package never mutates MODULE_BY_DIALECT. Consumers of the
    # view use membership, length and iteration over the same ordered values.
    revised = after.value
    after.value = before.value
    try:
        return ast.dump(old) == ast.dump(new)
    finally:
        after.value = revised


def concrete(path, family):
    parts = PurePosixPath(path).parts
    return (len(parts) == 3 and parts[:2] == ("sqlglot", family)
            and parts[-1].endswith(".py") and parts[-1] not in {"__init__.py", "base.py", "dialect.py"})


def target(node):
    if isinstance(node, ast.AnnAssign):
        return node.target
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        return node.targets[0]
    return None


def named(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def expression_name(node):
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "exp":
        return node.attr
    return None


def token_name(node):
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "TokenType":
        return node.attr
    return None


def inert(node, known_expression, callbacks=(), factories=()):
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Lambda):
        return all(value is None or isinstance(value, ast.Constant)
                   for value in (*node.args.defaults, *node.args.kw_defaults))
    if isinstance(node, ast.Attribute) and node.attr == "from_arg_list":
        return known_expression(expression_name(node.value))
    if isinstance(node, (ast.Name, ast.Attribute)):
        return ast.unparse(node) in callbacks
    if isinstance(node, ast.Call) and not node.keywords and len(node.args) == 1:
        factory = ast.unparse(node.func)
        if factory not in factories:
            return False
        arg = node.args[0]
        if factory == "rename_func":
            return isinstance(arg, ast.Constant) and type(arg.value) is str
        return isinstance(arg, ast.List) and all(ast.unparse(item) in callbacks for item in arg.elts)
    return False


def fromkeys(node):
    if not (isinstance(node, ast.Call) and ast.unparse(node.func) == "dict.fromkeys"
            and len(node.args) == 2 and not node.keywords and isinstance(node.args[0], ast.Tuple)):
        return None
    return [(key, node.args[1]) for key in node.args[0].elts]


def generator_spread(node):
    if isinstance(node, ast.Attribute) and ast.unparse(node) == "generator.Generator.TRANSFORMS":
        return ast.dump(node), set()
    if not isinstance(node, ast.DictComp) or len(node.generators) != 1:
        return None
    loop = node.generators[0]
    if not (pure_filter(loop, node) and len(loop.ifs) == 1 and isinstance(loop.ifs[0].ops[0], ast.NotIn)
            and isinstance(loop.ifs[0].comparators[0], ast.Set) and isinstance(loop.iter, ast.Call)
            and isinstance(loop.iter.func, ast.Attribute) and loop.iter.func.attr == "items"
            and not loop.iter.args and not loop.iter.keywords):
        return None
    base = loop.iter.func.value
    if ast.unparse(base) != "generator.Generator.TRANSFORMS":
        return None
    keys = {expression_name(item) for item in loop.ifs[0].comparators[0].elts}
    return (ast.dump(base), keys) if None not in keys else None


def lookups(table, field, allow_fromkeys):
    """Separate explicit data from opaque spreads, preserving spread precedence."""
    entries, spreads, order = defaultdict(list), [], []
    for key, value in zip(table.keys, table.values):
        pairs = fromkeys(value) if key is None and allow_fromkeys else None
        if pairs is None and key is None:
            spread = generator_spread(value) if field == "TRANSFORMS" else None
            spreads.append(spread if spread is not None else (ast.dump(value), set()))
            continue
        for k, v in pairs if pairs is not None else ((key, value),):
            if field == "TRANSFORMS":
                name = expression_name(k)
            elif field == "RANGE_PARSERS":
                name = token_name(k)
            else:
                name = k.value if isinstance(k, ast.Constant) and type(k.value) is str else None
            if name is None:
                return None, (), ()
            order.append(name)
            entries[name].append((len(spreads), ast.dump(k), ast.dump(v), v))
    return entries, spreads, order


def table_changes(path, original, updated, known_expression, transform_callbacks=(), known_tokens=()):
    """Accept only changes to inert entries in existing dialect dispatch tables."""
    shared = path == "sqlglot/generator.py"
    fields = TABLES["generators"] if shared else next(
        (fields for family, fields in TABLES.items() if concrete(path, family)), None)
    if fields is None or not same_directive_lines(original, updated):
        return {}
    old, new = ast.parse(original), ast.parse(updated)
    bound = {node.id for node in ast.walk(old) if isinstance(node, ast.Name)
             and isinstance(node.ctx, (ast.Store, ast.Del))}
    imports = {alias.asname or alias.name for alias in ast.walk(old) if isinstance(alias, ast.alias)}
    allow_fromkeys = "dict" not in bound | imports
    if "exp" in bound or not any(
        isinstance(node, ast.ImportFrom) and node.module == "sqlglot" and not node.level
        and any(alias.name == "exp" and alias.asname is None for alias in node.names) for node in old.body
    ):
        known_expression = lambda name: False
    if "TokenType" in bound or not any(
        isinstance(node, ast.ImportFrom) and node.module in {"sqlglot.tokens", "sqlglot.tokenizer_core"}
        and not node.level and any(alias.name == "TokenType" and alias.asname is None for alias in node.names)
        for node in old.body
    ):
        known_tokens = ()
    # SQLGlot's closure factories bind callbacks at import and invoke them only
    # when chosen by the generator. New helper definitions must have no other effect.
    old_names = bound | imports | {node.id for node in ast.walk(old) if isinstance(node, ast.Name)}
    old_names.update(node.name for node in ast.walk(old) if isinstance(node, (ast.ClassDef, ast.FunctionDef)))
    postponed = any(
        isinstance(node, ast.ImportFrom) and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names) for node in old.body
    )
    helpers = {}
    if concrete(path, "generators") and postponed:
        retained = []
        for node in new.body:
            if (isinstance(node, ast.FunctionDef) and node.name.startswith("_") and not node.name.startswith("__")
                    and node.name not in old_names and node.name not in helpers and not node.decorator_list
                    and not getattr(node, "type_params", ()) and not node.args.defaults
                    and not any(value is not None for value in node.args.kw_defaults)):
                helpers[node.name] = node.lineno
            else:
                retained.append(node)
        new.body = retained
    existing = {node.name: node.lineno for node in old.body
                if isinstance(node, ast.FunctionDef) and node.name not in bound}
    # Some pinned callbacks are already-bound closures rather than def statements.
    for node in old.body:
        if (isinstance(node, ast.Assign) and isinstance(lhs := target(node), ast.Name)
                and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "binary_range_parser"):
            existing[lhs.id] = node.lineno
    revised_defs = {
        name: node.lineno for node in new.body
        if (name := node.name if isinstance(node, ast.FunctionDef) else named(target(node))) in existing
    }
    revised_defs.update(helpers)
    factories = set()
    attrs = set()
    if "transforms" not in bound and any(
        isinstance(node, ast.ImportFrom) and node.module == "sqlglot" and not node.level
        and any(alias.name == "transforms" and alias.asname is None for alias in node.names) for node in old.body
    ):
        attrs.update(transform_callbacks)
        factories.add("transforms.preprocess")
    if "rename_func" not in bound and any(
        isinstance(node, ast.ImportFrom) and node.module == "sqlglot.dialects.dialect" and not node.level
        and any(alias.name == "rename_func" and alias.asname is None for alias in node.names) for node in old.body
    ):
        factories.add("rename_func")

    def declarations(tree):
        found = {}
        for cls in tree.body:
            if not isinstance(cls, ast.ClassDef) or cls.decorator_list or cls.keywords:
                continue
            for stmt in cls.body:
                lhs = target(stmt)
                name = lhs.id if isinstance(lhs, ast.Name) else None
                if name not in fields:
                    continue
                key = cls.name, name
                found[key] = stmt if key not in found else None
        return found

    left, right = declarations(old), declarations(new)
    if left.keys() != right.keys():
        return {}
    edits, used_helpers = {}, set()
    for key, a in left.items():
        b = right[key]
        if a is None or b is None or ast.dump(a) == ast.dump(b):
            continue
        if not isinstance(a.value, ast.Dict) or not isinstance(b.value, ast.Dict):
            return {}
        field = key[1]
        if shared and (key[0] != "Generator" or len(a.value.keys) != len(b.value.keys)
                       or any((x is None) != (y is None) or x is not None and ast.dump(x) != ast.dump(y)
                              for x, y in zip(a.value.keys, b.value.keys))):
            return {}
        prior, spreads, order = lookups(a.value, field, allow_fromkeys)
        revised, new_spreads, new_order = lookups(b.value, field, allow_fromkeys)
        if prior is None or revised is None or [kind for kind, _ in spreads] != [kind for kind, _ in new_spreads]:
            return {}
        # A truthiness branch on a dictionary can observe an empty table.
        if not (spreads or (prior and revised)):
            return {}
        def signature(table, name):
            return [entry[:3] for entry in table.get(name, ())]

        changes = {name for name in prior.keys() | revised.keys()
                   if signature(prior, name) != signature(revised, name)}
        for (_, excluded), (_, new_excluded) in zip(spreads, new_spreads):
            changes.update(excluded ^ new_excluded)
        if not changes:
            return {}
        if [name for name in order if name not in changes] != [name for name in new_order if name not in changes]:
            return {}
        values = ((prior, a, existing), (revised, b, revised_defs))
        values = [(table, attrs | {name for name, line in defs.items() if line < decl.lineno})
                  for table, decl, defs in values]
        for name in changes:
            if field == "TRANSFORMS" and not known_expression(name):
                return {}
            if field == "RANGE_PARSERS" and name not in known_tokens:
                return {}
            if shared:
                # Preserve the dispatch shape and eager evaluation. Only an
                # existing lambda's deferred behavior may change at this key.
                x, y = prior.get(name, ()), revised.get(name, ())
                if (len(x) != 1 or len(y) != 1 or not isinstance(x[0][3], ast.Lambda)
                        or not isinstance(y[0][3], ast.Lambda) or ast.dump(x[0][3].args) != ast.dump(y[0][3].args)):
                    return {}
            for table, callbacks in values:
                for _, _, _, value in table.get(name, ()):
                    if not inert(value, known_expression, callbacks, factories):
                        return {}
                    used_helpers.update(node.id for node in ast.walk(value)
                                        if isinstance(node, ast.Name) and node.id in helpers)
        edits.setdefault(field, set()).update(changes)
        b.value = a.value
    if used_helpers != helpers.keys() or ast.dump(old) != ast.dump(new):
        return {}
    if helpers:
        edits["MODULE_ATTRIBUTE"] = set(helpers)
    return edits


def assigned_to(node, parents, field):
    parent = parents.get(node)
    return named(target(parent)) == field and parent.value is node


def pure_filter(loop, comp):
    if not (isinstance(comp, ast.DictComp) and isinstance(loop.target, ast.Tuple)
            and len(loop.target.elts) == 2 and all(isinstance(e, ast.Name) for e in (*loop.target.elts, comp.key, comp.value))
            and [named(e) for e in loop.target.elts] == [comp.key.id, comp.value.id]
            and len(comp.generators) == 1 and not loop.is_async):
        return False
    return all(isinstance(check, ast.Compare) and isinstance(check.left, ast.Name)
               and check.left.id == comp.key.id and len(check.ops) == 1
               and isinstance(check.ops[0], (ast.NotEq, ast.NotIn))
               and all(not isinstance(n, (ast.Call, ast.Subscript, ast.NamedExpr))
                       for n in ast.walk(check.comparators[0])) for check in loop.ifs)


def transmitted(node, parents, field):
    parent = parents.get(node)
    if isinstance(parent, ast.Dict):
        if not any(key is None and value is node for key, value in zip(parent.keys, parent.values)):
            return False
        if assigned_to(parent, parents, field):
            return True
        method = parents.get(parent)
        call = parents.get(method)
        loop = parents.get(call)
        comp = parents.get(loop)
        return (isinstance(method, ast.Attribute) and method.attr == "items"
                and isinstance(loop, ast.comprehension) and loop.iter is call
                and pure_filter(loop, comp) and assigned_to(comp, parents, field))
    if isinstance(parent, ast.Attribute) and parent.attr == "copy":
        call = parents.get(parent)
        return isinstance(call, ast.Call) and assigned_to(call, parents, field)
    if not isinstance(parent, ast.Attribute) or parent.attr != "items":
        return False
    call = parents.get(parent)
    loop = parents.get(call)
    comp = parents.get(loop)
    if not (isinstance(loop, ast.comprehension) and loop.iter is call and pure_filter(loop, comp)):
        return False
    owner = parents.get(comp)
    return assigned_to(comp, parents, field) or (
        isinstance(owner, ast.Dict) and assigned_to(owner, parents, field)
        and any(k is None and v is comp for k, v in zip(owner.keys, owner.values))
    )


class DialectTableReads:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        file = Path(__file__).resolve().parents[1] / "maps/dialect-reads.json.gz"
        self.records = json.loads(gzip.decompress(file.read_bytes())) if file.is_file() else {}
        self.observed, self.traced = defaultdict(int), defaultdict(int)
        for identity, record in self.records.items():
            number = analyzer.num.get(identity)
            if number is not None:
                mask = 1 << number
            else:
                mask = 0
                for test in analyzer.tests:
                    if test["file"] == identity and test["id"] not in self.records:
                        mask |= 1 << test["number"]
            for field, keys in record.items():
                self.traced[field] |= mask
                for key in keys:
                    self.observed[field, key] |= mask

    @lru_cache(maxsize=16)
    def known_expression(self, name):
        return bool(name and not name.startswith("_") and self.analyzer.expr_owners("", {name}))

    @lru_cache(maxsize=1)
    def transform_callbacks(self):
        tree, _ = self.analyzer.old_tree_and_lines("sqlglot/transforms.py")
        return {f"transforms.{node.name}" for node in tree.body if isinstance(node, ast.FunctionDef)}

    @lru_cache(maxsize=1)
    def known_tokens(self):
        tree, _ = self.analyzer.old_tree_and_lines("sqlglot/tokenizer_core.py")
        tokens = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TokenType")
        return {lhs.id for node in tokens.body if isinstance(lhs := target(node), ast.Name)
                and isinstance(node.value, ast.Call) and ast.unparse(node.value) == "auto()"}

    def generator_methods(self, path, old, new):
        if not concrete(path, "generators"):
            return {}, 0
        if not same_directive_lines(old, new):
            return {}, 0
        before, after = ast.parse(old), ast.parse(new)
        names = [node for node in ast.walk(before) if isinstance(node, ast.Name)]
        assignments = {node.id for node in names if isinstance(node.ctx, (ast.Store, ast.Del))}
        definitions = {node.name for node in ast.walk(before) if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
        imports = {alias.asname or alias.name for alias in ast.walk(before) if isinstance(alias, ast.alias)}
        old_names = definitions | imports | {node.id for node in names}
        builtin_frozenset = "frozenset" not in assignments | definitions | imports
        literal_bindings = {}
        body = []
        for node in after.body:
            lhs = target(node)
            literal = (isinstance(lhs, ast.Name) and lhs.id.startswith("_") and not lhs.id.startswith("__")
                       and lhs.id not in old_names and lhs.id not in literal_bindings
                       and isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                       and ast.unparse(node.value.func) == "frozenset" and builtin_frozenset
                       and not node.value.keywords and len(node.value.args) == 1
                       and isinstance(node.value.args[0], ast.Constant)
                       and type(node.value.args[0].value) is str)
            if literal:
                literal_bindings[lhs.id] = node.lineno
            else:
                body.append(node)
        after.body = body
        can_wrap = "unsupported_args" not in assignments | definitions and any(
            isinstance(node, ast.ImportFrom) and node.module == "sqlglot.generator" and not node.level
            and any(alias.name == "unsupported_args" and alias.asname is None for alias in node.names)
            for node in before.body
        )
        used = set()
        for cls in after.body:
            if not isinstance(cls, ast.ClassDef):
                continue
            for method in cls.body:
                if not isinstance(method, ast.FunctionDef) or not method.name.endswith("_sql"):
                    continue
                if method.name not in old_names:
                    used.update(node.id for node in ast.walk(method) if isinstance(node, ast.Name)
                                and node.id in literal_bindings and literal_bindings[node.id] < cls.lineno)
                    if can_wrap and len(method.decorator_list) == 1:
                        decorator = method.decorator_list[0]
                        if (isinstance(decorator, ast.Call) and ast.unparse(decorator.func) == "unsupported_args"
                                and not decorator.keywords
                                and all(isinstance(arg, ast.Constant) and type(arg.value) is str for arg in decorator.args)):
                            method.decorator_list.clear()
        if used != literal_bindings.keys():
            return {}, 0
        methods = added_methods(before, after)
        if not methods or any(name.startswith("_") or not name.endswith("_sql") for name in methods):
            return {}, 0
        a = self.analyzer
        self.known_expression("Expression")
        classes = {name for name in a._expr_classes if name.lower() + "_sql" in methods}
        if {name.lower() + "_sql" for name in classes} != methods:
            return {}, 0
        mask = 0
        for name in methods:
            mask |= a.state_refs.mask(name)
        edits = {"TRANSFORMS": classes, "GENERATOR_METHOD": methods}
        if used:
            edits["MODULE_ATTRIBUTE"] = used
        return edits, mask

    @lru_cache(maxsize=None)
    def observer_sites(self, field):
        a = self.analyzer
        sites = {}
        by_key = defaultdict(int)
        names = (field, "expression_metadata", "HIVE_EXPRESSION_METADATA") if field == "EXPRESSION_METADATA" else (
            (field, "_dispatch") if field == "TRANSFORMS" else (field,)
        )
        for name in names:
            for path, lines in a.state_refs.refs.get(name, {}).items():
                if path.startswith("tests/"):
                    sites.setdefault(path, set()).update(lines)
                    continue
                tree, _ = a.old_tree_and_lines(path)
                parents = {node: parent for parent in ast.walk(tree) for node in ast.iter_child_nodes(parent)}
                safe = set()
                unsafe = set()
                monitored = {expr for expr, fields in SITES.get(path, {}).items() if field in fields}
                for node, parent in parents.items():
                    if not isinstance(node, (ast.Name, ast.Attribute)) or named(node) != name:
                        continue
                    if not isinstance(node.ctx, ast.Load):
                        continue
                    grandparent = parents.get(parent)
                    expr = ast.unparse(grandparent) if isinstance(grandparent, ast.Call) else ""
                    passive = transmitted(node, parents, field)
                    passive |= path == "sqlglot/parser.py" and ast.unparse(parent) == "functions = self.FUNCTIONS"
                    passive |= path == "sqlglot/parser.py" and isinstance(parent, ast.Call) and (
                        isinstance(parent.func, ast.Attribute) and parent.func.attr == "_match_texts"
                        and parent.args == [node] and field == "PROPERTY_PARSERS"
                    )
                    if path == "sqlglot/parser.py" and field == "RANGE_PARSERS":
                        passive |= ast.unparse(parent) in {
                            "self._match_set(self.RANGE_PARSERS)",
                            "self.RANGE_PARSERS[self._prev.token_type]",
                            "self._curr.token_type in self.RANGE_PARSERS",
                        }
                    passive |= path == "sqlglot/generator.py" and isinstance(parent, ast.Call) and (
                        ast.unparse(parent) == "dict(cls.TRANSFORMS)"
                    )
                    passive |= path == "sqlglot/dialects/dialect.py" and expr == "gen_cls.TRANSFORMS.pop(part, None)"
                    passive |= path == "sqlglot/optimizer/annotate_types.py" and ast.unparse(parent) == (
                        "expression_metadata or dialect.EXPRESSION_METADATA"
                    )
                    passive |= (path == "sqlglot/optimizer/annotate_types.py" and isinstance(parent, ast.keyword)
                                and parent.arg == "expression_metadata" and isinstance(grandparent, ast.Call)
                                and named(grandparent.func) == "TypeAnnotator")
                    if (field in TABLES["parsers"] and isinstance(parent, ast.Subscript)
                            and parent.value is node and isinstance(parent.slice, ast.Constant)
                            and type(parent.slice.value) is str):
                        by_key[parent.slice.value] |= a._lines_and_indirect_mask(path, (node.lineno, node.end_lineno))
                        passive = True
                    destination = safe if passive or expr in monitored else unsafe
                    destination.update((node.lineno, node.end_lineno))
                if field == "EXPRESSION_METADATA":
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ImportFrom) and any(alias.name == name for alias in node.names):
                            safe.add(node.lineno)
                sites.setdefault(path, set()).update(set(lines) - (safe - unsafe))
        mask = 0
        for path, lines in sites.items():
            mask |= a._lines_and_indirect_mask(path, lines)
        return mask, by_key

    @lru_cache(maxsize=None)
    def observers(self, field, keys=()):
        a = self.analyzer
        mask = a.state_refs.mask("*") | a.location_observers() | a.doc_observers()
        if field == "MODULE_ATTRIBUTE":
            for key in keys:
                mask |= a.state_refs.mask(key)
            for path, lines in a.state_refs.refs.get("*module", {}).items():
                if path == "sqlglot/optimizer/__init__.py":
                    # Its literal lazy-load table names only optimizer modules.
                    continue
                if path == "sqlglot/serde.py":
                    tree, _ = a.old_tree_and_lines(path)
                    lines = set(lines) - {node.lineno for node in ast.walk(tree)
                                          if isinstance(node, ast.Call) and ast.unparse(node) == "getattr(module, class_name)"}
                mask |= a._lines_and_indirect_mask(path, lines)
        fixed, by_key = self.observer_sites(field)
        mask |= fixed
        for key in keys or by_key:
            mask |= by_key.get(key, 0)
        return mask

    def unrelated(self, edits):
        if not self.records or not edits:
            return 0
        mask = -1
        for field, keys in edits.items():
            keep = self.observers(field, tuple(sorted(keys)))
            for key in keys:
                keep |= self.observed[field, key]
            mask &= self.traced[field] & ~keep
        return mask
