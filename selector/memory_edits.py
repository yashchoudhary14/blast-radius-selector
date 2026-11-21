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


def same_memory_workload(analyzer, edits, bit):
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
