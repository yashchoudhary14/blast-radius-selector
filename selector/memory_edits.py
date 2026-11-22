"""Prove that unread constants cannot change the generator allocation loop."""

import ast
from copy import copy

from .core_headers import inert_header
from .dialect_tables import concrete, same_directive_lines
from .query_bindings import bindings, declaration
from .query_cases import frame_layout
from .scopes import snapshot
from .source_equivalence import compile_source


def scalar_kind(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, str, bool, type(None)):
        return type(node.value)
    if (isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd))
            and isinstance(node.operand, ast.Constant) and type(node.operand.value) is int):
        return int
    return None


def container_kind(node):
    kind = scalar_kind(node)
    if kind is None and isinstance(node, (ast.Tuple, ast.Set)) and all(scalar_kind(item) for item in node.elts):
        return type(node)
    return kind


def constants(old, new, path):
    """Undo value replacements and verify every other AST node is unchanged."""
    revised = ast.parse(new, path)
    fields = set()
    dialect = any(concrete(path, family) for family in ("dialects", "generators", "parsers"))
    postponed = any(isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__"
                    and any(alias.name == "annotations" for alias in stmt.names) for stmt in old.body)

    def replace(left, right, *, class_body=False):
        if len(left) != len(right):
            return
        for before, after in zip(left, right):
            if (dialect and isinstance(before, ast.ClassDef) and isinstance(after, ast.ClassDef)
                    and not before.decorator_list and not before.keywords):
                replace(before.body, after.body, class_body=True)
            if type(before) is not type(after) or not isinstance(before, (ast.Assign, ast.AnnAssign)):
                continue
            name = declaration(before, postponed)
            if (not name or name != declaration(after, postponed)
                    or ast.dump(before.value) == ast.dump(after.value)):
                continue
            if class_body:
                # A class literal can be consumed by its metaclass or a derived
                # declaration. Keep those readers even when they only run at import.
                kind_of = container_kind if concrete(path, "parsers") else scalar_kind
                kind = kind_of(before.value)
                if kind is None or kind is not kind_of(after.value):
                    continue
            after.value = before.value
            fields.add(name)

    replace(old.body, revised.body)
    return fields if fields and ast.dump(old) == ast.dump(revised) else set()


def dormant_bodies(analyzer, path, tree, compiled, before, after, bit):
    from .analyze import _get_func_node

    changed = [name for name, row in before.items() if row[:3] != after[name][:3]]
    if not changed:
        return False
    old_tree, _ = analyzer.old_tree_and_lines(path)
    for name in changed:
        fn = _get_func_node(old_tree, name)
        updated = _get_func_node(tree, name)
        if (fn is None or updated is None or fn.decorator_list or getattr(fn, "type_params", ())
                or ast.get_docstring(fn, clean=False) != ast.get_docstring(updated, clean=False)
                or analyzer._lines_and_indirect_mask(path, range(fn.body[0].lineno, fn.end_lineno + 1)) & bit):
            return False
        if before[name][1] != after[name][1]:
            old_code, _ = analyzer.old_compiled(path)
            if ((analyzer.signature_observers(path) | analyzer.state_refs.mask("signature")) & bit
                    or inert_header(fn, updated, old_code.co_flags, compiled.co_flags) != "default"):
                return False
            # Verify that the restored literal defaults account for every
            # header change, including parameter and result annotations.
            old_header, new_header = copy(fn), copy(updated)
            old_header.body = new_header.body = []
            new_header.args = copy(updated.args)
            new_header.args.defaults = fn.args.defaults
            new_header.args.kw_defaults = fn.args.kw_defaults
            if ast.dump(old_header) != ast.dump(new_header):
                return False
    return (not analyzer.old_compiled(path)[1]
            and analyzer.old_layout(path, locals=False) == frame_layout(compiled, locals=False))


def same_memory_workload(analyzer, edits, bit, revisions):
    if len(edits) != 1:
        return False
    path, revised = next(iter(edits.items()))
    if (not revised or not path.startswith("sqlglot/") or not path.endswith(".py")
            or path.endswith("/__init__.py") or analyzer.index["reads"].get(path, 0) & bit):
        return False
    file = analyzer.base / path
    if not file.is_file() or path in edits.modes and edits.modes[path] != file.stat().st_mode & 0o777:
        return False
    before = file.read_bytes()
    if analyzer.location_observers(all_dynamic=True) & bit:
        return False
    old_funcs, old_scopes = analyzer.old(path)
    revision = revisions.get(path)
    if revision is None:
        tree = ast.parse(revised, path)
        compiled, _ = compile_source(revised, path)
        new_funcs, new_scopes = snapshot(tree, revised)
    else:
        tree, compiled, new_funcs, new_scopes = revision
    if old_scopes == new_scopes and old_funcs.keys() == new_funcs.keys():
        # Constant and module binding changes would alter this snapshot.
        # Only a body untouched by setup, warmup and the allocation window
        # can be ignored here.
        return (dormant_bodies(analyzer, path, tree, compiled, old_funcs, new_funcs, bit)
                and same_directive_lines(before, revised))
    original, _ = analyzer.old_tree_and_lines(path)
    fields = constants(original, revised, path)
    if not fields:
        binding_change = bindings(before, revised, path)
        if binding_change is None:
            return False
        values, added = binding_change
        if values or not added or any(not name.startswith("_") for name in added):
            return False
        fields = added
    # These assignments execute once, before the allocation loop. Their
    # values cannot change class construction, warmup or the measured loop
    # without a reader in one of those phases.
    return (all(not analyzer.state_refs.mask(field) & bit for field in fields | {"*", "*module"})
            and same_directive_lines(before, revised))
