"""Prove that a patch leaves the generator allocation workload unchanged."""

import ast
from copy import copy

from .core_headers import inert_header
from .dialect_tables import concrete, same_directive_lines
from .query_bindings import bindings, declaration
from .query_cases import frame_layout
from .scopes import FUNCTION, snapshot
from .source_equivalence import compile_source, same_code


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


def constants(old, revised, path):
    """Undo value replacements and verify every other AST node is unchanged."""
    restored = []
    fields = set()
    dialect = any(concrete(path, family) for family in ("dialects", "generators", "parsers"))
    kind_of = container_kind if concrete(path, "parsers") else scalar_kind
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
                # Class literals can be consumed during class construction.
                kind = kind_of(before.value)
                if kind is None or kind is not kind_of(after.value):
                    continue
            restored.append((after, after.value))
            after.value = before.value
            fields.add(name)

    try:
        replace(old.body, revised.body)
        return fields if fields and ast.dump(old) == ast.dump(revised) else set()
    finally:
        for stmt, value in restored:
            stmt.value = value


def parser_methods_only(old, new):
    """Recognize edits confined to ordinary methods of unchanged parser classes."""
    # The caller has already compared the ordered scope snapshots.
    for before, after in zip(old.body, new.body):
        if not isinstance(before, ast.ClassDef):
            if ast.dump(before) != ast.dump(after):
                return False
            continue
        if before.decorator_list or before.keywords:
            return False
        if any(stmt.decorator_list for stmt in after.body if isinstance(stmt, FUNCTION)):
            return False
    return True


def dormant_bodies(analyzer, path, old, tree, compiled, before, after, bit):
    from .analyze import _get_func_node

    changed = [name for name, row in before.items() if row[:3] != after[name][:3]]
    if not changed:
        return False
    old_code, diagnostics = analyzer.old_compiled(path)
    if diagnostics:
        return False
    for name in changed:
        fn = _get_func_node(old, name)
        updated = _get_func_node(tree, name)
        if (fn is None or updated is None or fn.decorator_list or getattr(fn, "type_params", ())
                or ast.get_docstring(fn, clean=False) != ast.get_docstring(updated, clean=False)
                or analyzer._lines_and_indirect_mask(path, range(fn.body[0].lineno, fn.end_lineno + 1)) & bit):
            return False
        if before[name][1] != after[name][1]:
            if ((analyzer.signature_observers(path) | analyzer.state_refs.mask("signature")) & bit
                    or inert_header(fn, updated, old_code.co_flags, compiled.co_flags) != "default"):
                return False
            # Restoring literal defaults must account for the entire header.
            old_header, new_header = copy(fn), copy(updated)
            old_header.body = new_header.body = []
            new_header.args = copy(updated.args)
            new_header.args.defaults = fn.args.defaults
            new_header.args.kw_defaults = fn.args.kw_defaults
            if ast.dump(old_header) != ast.dump(new_header):
                return False
    return analyzer.old_layout(path, locals=False) == frame_layout(compiled, locals=False)


def unchanged_module_workload(analyzer, path, source, bit, revision, *, parser_only):
    if not source or not path.startswith("sqlglot/") or not path.endswith(".py") or path.endswith("/__init__.py"):
        return False
    original = (analyzer.base / path).read_bytes()
    if not same_directive_lines(original, source):
        return False

    old_funcs, old_scopes = analyzer.old(path)
    if revision is None:
        tree = ast.parse(source, path)
        compiled, diagnostics = compile_source(source, path)
        if diagnostics:
            return False
        new_funcs, new_scopes = snapshot(tree, source)
    else:
        tree, compiled, new_funcs, new_scopes = revision
    old, _ = analyzer.old_tree_and_lines(path)
    if old_scopes == new_scopes and old_funcs.keys() == new_funcs.keys():
        if parser_only and not parser_methods_only(old, tree):
            return False
        if all(row[:3] == new_funcs[name][:3] for name, row in old_funcs.items()):
            before, diagnostics = analyzer.old_compiled(path)
            # Exact code equality also preserves the frame layout of an
            # unchanged parser accompanying a proved dormant method edit.
            return not diagnostics and same_code(before, compiled)
        return dormant_bodies(analyzer, path, old, tree, compiled, old_funcs, new_funcs, bit)
    if parser_only:
        return False

    fields = constants(old, tree, path)
    if not fields:
        binding_change = bindings(original, source, path)
        if binding_change is None:
            return False
        values, added = binding_change
        if values or not added or any(not name.startswith("_") for name in added):
            return False
        fields = added
    # Include import-time references. A copied value can be consumed before
    # warmup even when no reader executes during the measurement itself.
    return all(not analyzer.state_refs.mask(field) & bit for field in fields | {"*"})


def same_memory_workload(analyzer, edits, bit, revisions, isolated_test_paths):
    parser_only = len(edits) > 1
    if len(edits) == 1:
        path = next(iter(edits))
        if path in isolated_test_paths or not path.startswith("sqlglot/"):
            return False
    elif not edits or any(path not in isolated_test_paths and not concrete(path, "parsers") for path in edits):
        return False
    if (analyzer.location_observers(all_dynamic=True) | analyzer.state_refs.mask("*module")) & bit:
        return False
    for path, source in edits.items():
        if analyzer.index["reads"].get(path, 0) & bit:
            return False
        file = analyzer.base / path
        if not file.is_file() or (path in edits.modes and edits.modes[path] != file.stat().st_mode & 0o777):
            return False
        if path in isolated_test_paths:
            if isolated_test_paths[path] & bit:
                return False
        elif not unchanged_module_workload(analyzer, path, source, bit, revisions.get(path),
                                           parser_only=parser_only):
            return False
    return True
