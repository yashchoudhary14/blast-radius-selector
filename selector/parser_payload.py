"""Recognize ParseError payload edits invisible to the parser limit test."""

import ast

from .dialect_tables import same_directive_lines


ERRORS = "sqlglot/errors.py"


def payload(tree):
    cls = next((s for s in tree.body if isinstance(s, ast.ClassDef) and s.name == "ParseError"), None)
    if cls is None:
        return None
    fn = next((s for s in cls.body if isinstance(s, ast.FunctionDef) and s.name == "new"), None)
    if fn is None or len(fn.body) != 1 or not isinstance(fn.body[0], ast.Return):
        return None
    call = fn.body[0].value
    if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != "cls"
            or call.keywords or len(call.args) != 2 or not isinstance(call.args[0], ast.Name)
            or call.args[0].id != "message"):
        return None
    rows = call.args[1]
    if isinstance(rows, ast.List) and len(rows.elts) == 1 and isinstance(rows.elts[0], ast.Dict):
        return rows.elts[0]
    return None


def inert_value(node, param):
    if isinstance(node, ast.Name):
        return node.id == param
    if isinstance(node, ast.Constant):
        return type(node.value) in (str, int, bool, type(None))
    if not isinstance(node, ast.IfExp) or not isinstance(node.test, ast.Compare):
        return False
    test = node.test
    return (isinstance(test.left, ast.Name) and test.left.id == param and len(test.ops) == 1
            and isinstance(test.ops[0], (ast.Is, ast.IsNot))
            and isinstance(test.comparators[0], ast.Constant) and test.comparators[0].value is None
            and not isinstance(node.body, ast.IfExp) and not isinstance(node.orelse, ast.IfExp)
            and inert_value(node.body, param) and inert_value(node.orelse, param))


def unchanged_limit_assertions(analyzer, edits, bit):
    if edits.keys() != {ERRORS} or not edits[ERRORS] or analyzer.index["reads"].get(ERRORS, 0) & bit:
        return False
    file = analyzer.base / ERRORS
    if ERRORS in edits.modes and edits.modes[ERRORS] != file.stat().st_mode & 0o777:
        return False
    before, after = file.read_bytes(), edits[ERRORS]
    if not same_directive_lines(before, after):
        return False
    old, new = ast.parse(before), ast.parse(after)
    a, b = payload(old), payload(new)
    if a is None or b is None or ast.dump(a) == ast.dump(b):
        return False
    if not all(isinstance(key, ast.Constant) and type(key.value) is str for key in b.keys):
        return False
    keys = {key.value for key in b.keys}
    if len(keys) != len(b.keys) or keys != {key.value for key in a.keys}:
        return False
    if not all(inert_value(value, key.value) for key, value in zip(b.keys, b.values)):
        return False
    # The pinned limit test observes the exception type and message. The
    # parser can concatenate messages and aggregate these records but does
    # not use a payload field to select a parse path or form the message.
    b.keys, b.values = a.keys, a.values
    return ast.dump(old) == ast.dump(new)
