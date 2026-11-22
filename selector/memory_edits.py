"""Prove that unread constants cannot change the generator allocation loop."""

import ast

from .dialect_tables import concrete, same_directive_lines, target
from .query_bindings import bindings, declaration, reference_lines


def scalar_kind(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, str, bool, type(None)):
        return type(node.value)
    if (isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd))
            and isinstance(node.operand, ast.Constant) and type(node.operand.value) is int):
        return int
    return None


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
                kind = scalar_kind(before.value)
                if kind is None or kind is not scalar_kind(after.value):
                    continue
            after.value = before.value
            fields.add(name)

    replace(old.body, revised.body)
    return fields if fields and ast.dump(old) == ast.dump(revised) else set()


_PARSER_TABLES = frozenset({"FUNCTIONS", "FUNCTION_PARSERS", "PROPERTY_PARSERS", "RANGE_PARSERS"})


def _parser_table_lines(analyzer, path):
    if not concrete(path, "parsers"):
        return set()
    tree, _ = analyzer.old_tree_and_lines(path)
    lines = set()
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and not cls.decorator_list and not cls.keywords:
            for stmt in cls.body:
                lhs = target(stmt)
                if isinstance(lhs, ast.Name) and lhs.id in _PARSER_TABLES:
                    lines.update(range(stmt.lineno, (stmt.end_lineno or stmt.lineno) + 1))
    return lines


def _field_hits_bit(analyzer, field, bit):
    for reader_path in analyzer.state_refs.refs.get(field, ()):
        lines = reference_lines(analyzer, reader_path, field, preserve_binding=True)
        if not lines:
            continue
        direct = analyzer.line_hits(reader_path)
        imports = analyzer.indirect_hits.get(reader_path, {})
        parser_lines = _parser_table_lines(analyzer, reader_path)
        for ln in lines:
            if direct.get(ln, 0) & bit:
                return True
            if ln not in parser_lines and imports.get(ln, 0) & bit:
                return True
    return False


def is_safe_parser_edit(analyzer, path, revised, bit, edits):
    if not (revised and path.startswith("sqlglot/parsers/") and path.endswith(".py")
            and path != "sqlglot/parsers/__init__.py"):
        return False
    if analyzer.index["reads"].get(path, 0) & bit:
        return False
    file = analyzer.base / path
    if not file.is_file():
        return False
    mode = file.stat().st_mode & 0o777
    if edits.modes.get(path, mode) != mode:
        return False
    original = file.read_bytes()
    if not same_directive_lines(original, revised):
        return False
    try:
        old_tree = ast.parse(original, path)
        new_tree = ast.parse(revised, path)
    except (SyntaxError, ValueError):
        return False

    old_mod_stmts = [s for s in old_tree.body if not isinstance(s, ast.ClassDef)]
    new_mod_stmts = [s for s in new_tree.body if not isinstance(s, ast.ClassDef)]
    if len(old_mod_stmts) != len(new_mod_stmts):
        return False
    for s1, s2 in zip(old_mod_stmts, new_mod_stmts):
        if ast.dump(s1) != ast.dump(s2):
            return False

    old_classes = [s for s in old_tree.body if isinstance(s, ast.ClassDef)]
    new_classes = [s for s in new_tree.body if isinstance(s, ast.ClassDef)]
    if len(old_classes) != len(new_classes):
        return False
    for c1, c2 in zip(old_classes, new_classes):
        if c1.name != c2.name or c1.decorator_list or c2.decorator_list or c1.keywords or c2.keywords:
            return False
        if [ast.dump(b) for b in c1.bases] != [ast.dump(b) for b in c2.bases]:
            return False
        c1_non_fn = [s for s in c1.body if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))]
        c2_non_fn = [s for s in c2.body if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if len(c1_non_fn) != len(c2_non_fn):
            return False
        for s1, s2 in zip(c1_non_fn, c2_non_fn):
            if ast.dump(s1) != ast.dump(s2):
                return False
        for s in c2.body:
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if s.decorator_list:
                    return False
    return True


def safe_memory_parsers(analyzer, edits, isolated_test_paths, bit):
    for path, revised in edits.items():
        if path in isolated_test_paths:
            if isolated_test_paths[path] & bit:
                return False
            continue
        if not is_safe_parser_edit(analyzer, path, revised, bit, edits):
            return False
    return True


def same_memory_workload(analyzer, edits, bit, candidates=None):
    if (not analyzer.location_observers(all_dynamic=True) & bit
            and not analyzer.state_refs.mask("*") & bit
            and not analyzer.state_refs.mask("*module") & bit):
        if candidates is not None and safe_memory_parsers(analyzer, edits, candidates.isolated_test_paths, bit):
            return True
    if len(edits) != 1:
        return False
    path, revised = next(iter(edits.items()))
    if (not revised or not path.startswith("sqlglot/") or not path.endswith(".py")
            or path.endswith("/__init__.py") or analyzer.index["reads"].get(path, 0) & bit):
        return False
    file = analyzer.base / path
    if not file.is_file() or path in edits.modes and edits.modes[path] != file.stat().st_mode & 0o777:
        return False
    original, _ = analyzer.old_tree_and_lines(path)
    fields = constants(original, revised, path)
    before = file.read_bytes()
    if not same_directive_lines(before, revised):
        return False
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
    return (not analyzer.location_observers(all_dynamic=True) & bit
            and not analyzer.state_refs.mask("*") & bit
            and not analyzer.state_refs.mask("*module") & bit
            and all(not _field_hits_bit(analyzer, field, bit) for field in fields))
