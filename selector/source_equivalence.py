"""Prove that a source edit preserves the compiled program or changes one doc binding."""

import __future__
import ast
import dis
import marshal
import re
import tokenize
import warnings
from io import BytesIO
from types import CodeType


DIRECTIVE = re.compile(
    r"#\s*(?:!|.*?coding[=:]|(?:type|fmt|ruff|pyright|mypy|pylint|pragma|isort)\s*:|noqa\b)",
    re.IGNORECASE,
)


def compile_source(source, path):
    # A new compiler warning can fail import even when the bytecode is unchanged.
    with warnings.catch_warnings(record=True) as diagnostics:
        warnings.simplefilter("always")
        code = compile(source, path, "exec", dont_inherit=True)
    return code, tuple((warning.category, str(warning.message), warning.lineno) for warning in diagnostics)


def same_code(before, after):
    # CodeType equality omits some location information and compares numeric
    # constants by value. Marshal v2 retains columns, types and signed zero,
    # without making the comparison depend on process-specific reference counts.
    if marshal.dumps(before, 2) != marshal.dumps(after, 2):
        return False
    seen, reverse = {}, {}

    def same_sharing(left, right):
        a, b = id(left), id(right)
        if a in seen or b in reverse:
            return seen.get(a) == b and reverse.get(b) == a
        seen[a], reverse[b] = b, a
        if isinstance(left, CodeType):
            return same_sharing(left.co_consts, right.co_consts)
        if isinstance(left, (tuple, frozenset)):
            return all(same_sharing(x, y) for x, y in zip(left, right))
        return True

    return same_sharing(before, after)


def same_directives(before, after, insertion=None):
    def directives(source):
        return [(tok.start, tok.string) for tok in tokenize.tokenize(BytesIO(source).readline)
                if tok.type == tokenize.ENCODING or (tok.type == tokenize.COMMENT and DIRECTIVE.match(tok.string))]

    try:
        original = directives(before)
        if insertion:
            first, count = insertion
            original = [((line + count if line >= first else line, col), value)
                        for (line, col), value in original]
        return original == directives(after)
    except (SyntaxError, tokenize.TokenError, UnicodeError):
        return False


def _doc(node):
    if node.body and isinstance(node.body[0], ast.Expr):
        value = node.body[0].value
        if isinstance(value, ast.Constant) and type(value.value) is str:
            if value.value and ">>>" not in value.value:
                return value
    return None


def _outside(source, node):
    lines = source.splitlines(keepends=True)
    return (b"".join(lines[:node.lineno - 1]) + lines[node.lineno - 1][:node.col_offset],
            lines[node.end_lineno - 1][node.end_col_offset:] + b"".join(lines[node.end_lineno:]))


def _docs(tree):
    yield "", _doc(tree)
    for node in tree.body:
        if _plain_class(node):
            # An inherited __init_subclass__ or metaclass may consume the
            # new value while constructing a class, before it is exported.
            yield node.name, _doc(node)


def _plain_class(node):
    return (isinstance(node, ast.ClassDef)
            and not (node.bases or node.keywords or node.decorator_list or getattr(node, "type_params", ())))


def _doc_assignment(code, value):
    instructions = list(dis.get_instructions(code))
    for index, (load, store) in enumerate(zip(instructions, instructions[1:])):
        if (load.opname != "LOAD_CONST" or load.argval != value
                or store.opname != "STORE_NAME" or store.argval != "__doc__"):
            continue
        # A constant could also be used by a live instruction outside the
        # doc assignment, in which case clearing the slot would hide changes.
        if sum(i.opcode in dis.hasconst and i.arg == load.arg for i in instructions) == 1:
            positions = {load.offset // 2, store.offset // 2}
            # A module's annotation setup can inherit the first statement's
            # columns, including the closing quote of its docstring.
            if index and instructions[index - 1].opname == "SETUP_ANNOTATIONS":
                positions.add(instructions[index - 1].offset // 2)
            return load.arg, positions
    return None


def _without_doc_value(before, after, original, updated):
    old = _doc_assignment(before, original)
    new = _doc_assignment(after, updated)
    if old is None or old != new:
        return None
    slot, instructions = old
    for index, (left, right) in enumerate(zip(before.co_positions(), after.co_positions())):
        if left != right and (index not in instructions or left[:2] != right[:2]):
            return None
    constants = (*after.co_consts[:slot], original, *after.co_consts[slot + 1:])
    return after.replace(co_consts=constants, co_linetable=before.co_linetable)


def _class_body(code, name):
    matches = [(i, c) for i, c in enumerate(code.co_consts)
               if isinstance(c, CodeType) and c.co_qualname == name]
    return matches[0] if len(matches) == 1 else (None, None)


def docstring_only(old_source, source, old_tree, tree, before, after):
    """Accept one existing, doctest-free docstring with all live code intact."""
    old_docs = dict(_docs(old_tree))
    for name, revised in _docs(tree):
        original = old_docs.get(name)
        if original is None or revised is None or original.value == revised.value:
            continue
        if _outside(old_source, original) != _outside(source, revised):
            continue
        if not name:
            normalized = _without_doc_value(before, after, original.value, revised.value)
        else:
            old_index, old_body = _class_body(before, name)
            index, body = _class_body(after, name)
            if index is None or old_index != index:
                return False
            body = _without_doc_value(old_body, body, original.value, revised.value)
            if body is None:
                return False
            normalized = after.replace(co_consts=(*after.co_consts[:index], body, *after.co_consts[index + 1:]))
        return normalized is not None and same_code(before, normalized)
    return False


def _plain_function(tree, name):
    scope, _, member = name.rpartition(".")
    if scope:
        classes = [n for n in tree.body if _plain_class(n) and n.name == scope]
        if len(classes) != 1:
            return None
        tree = classes[0]
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == member]
    if len(functions) == 1:
        func = functions[0]
        if not func.decorator_list and not getattr(func, "type_params", ()):
            return func
    return None


def _signature_parts(func):
    args = func.args
    positional = args.posonlyargs + args.args
    parameters = positional + args.kwonlyargs
    parameters.extend(a for a in (args.vararg, args.kwarg) if a is not None)
    annotations = [a.annotation for a in parameters] + [func.returns]
    defaults = [(a.arg, d) for a, d in zip(positional[-len(args.defaults):], args.defaults)]
    defaults.extend((a.arg, d) for a, d in zip(args.kwonlyargs, args.kw_defaults))
    return annotations, defaults


def _isolated_change(old_source, source, old, new):
    return (old is not None and new is not None
            and (old.lineno, old.end_lineno) == (new.lineno, new.end_lineno)
            and ast.dump(old) != ast.dump(new)
            and _outside(old_source, old) == _outside(source, new))


def signature_metadata(old_source, source, old_tree, tree, name, before, after):
    """Accept one postponed annotation or one unused, scalar default value."""
    old = _plain_function(old_tree, name)
    revised = _plain_function(tree, name)
    if old is None or revised is None:
        return False
    old_annotations, old_defaults = _signature_parts(old)
    annotations, defaults = _signature_parts(revised)
    if any(d is not None and not isinstance(d, ast.Constant) for _, d in old_defaults):
        return False
    if before.co_flags & after.co_flags & __future__.annotations.compiler_flag:
        if any(_isolated_change(old_source, source, a, b)
               for a, b in zip(old_annotations, annotations)):
            return True
    for (parameter, a), (new_parameter, b) in zip(old_defaults, defaults):
        if (parameter == new_parameter and isinstance(a, ast.Constant) and isinstance(b, ast.Constant)
                and all(type(node.value) in (type(None), bool, int) for node in (a, b))
                and all(type(node.value) is not int or -5 <= node.value <= 256 for node in (a, b))
                and _isolated_change(old_source, source, a, b)
                and not any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                            and n.id in {parameter, "eval", "exec", "locals", "vars", "dir"}
                            for stmt in old.body for n in ast.walk(stmt))):
            return True
    return False


def line_insertion(before, after):
    """Locate a single insertion of complete lines into the original text."""
    if len(after) <= len(before):
        return None
    old, new = before.splitlines(keepends=True), after.splitlines(keepends=True)
    count = len(new) - len(old)
    if count <= 0:
        return None
    first = next((i for i, (left, right) in enumerate(zip(old, new)) if left != right), len(old))
    if old[first:] == new[first + count:]:
        return first + 1, count
    return None


def dead_statement(old, new, insertion):
    """Prove a single inert statement was inserted into an existing suite."""
    first, count = insertion
    for owner in ast.walk(new):
        for field, suite in ast.iter_fields(owner):
            if not isinstance(suite, list):
                continue
            for index, stmt in enumerate(suite):
                if not isinstance(stmt, ast.stmt) or not first <= stmt.lineno <= stmt.end_lineno < first + count:
                    continue
                # A nested class can capture function variables. A plain name
                # assignment has no binding effect in these suites.
                module_scope = owner is new or (isinstance(owner, ast.ClassDef) and owner in new.body)
                dead_branch = (module_scope and field == "body" and isinstance(stmt, ast.If) and not stmt.orelse
                               and isinstance(stmt.test, ast.Constant) and stmt.test.value is False
                               and len(stmt.body) == 1 and isinstance(stmt.body[0], ast.Assign)
                               and all(isinstance(t, ast.Name) for t in stmt.body[0].targets)
                               and isinstance(stmt.body[0].value, ast.Constant))
                # This raise cannot change local or closure bindings or mark
                # an otherwise synchronous function as a generator.
                dead_raise = (index > 0 and isinstance(suite[index - 1], ast.Return)
                              and isinstance(stmt, ast.Raise) and stmt.cause is None
                              and isinstance(stmt.exc, ast.Call) and isinstance(stmt.exc.func, ast.Name)
                              and not stmt.exc.keywords and all(isinstance(a, ast.Constant) for a in stmt.exc.args))
                # Preserve the first-statement docstring rule in every suite.
                if index and (dead_branch or dead_raise):
                    setattr(owner, field, suite[:index] + suite[index + 1:])
                    try:
                        return ast.dump(old) == ast.dump(new)
                    finally:
                        setattr(owner, field, suite)
    return False


def relocated_code(before, after, insertion):
    """Check live bytecode equality after moving source locations by one insertion."""
    first, count = insertion

    def move(line):
        return line + count if line is not None and line >= first else line

    def normalized(old, new):
        if (old.co_code != new.co_code or move(old.co_firstlineno) != new.co_firstlineno
                or old.co_names != new.co_names[:len(old.co_names)]
                or len(old.co_consts) != len(new.co_consts)):
            return None
        # With identical instructions, a new name after the old table's end is
        # unreachable by those instructions. The test observer mask still
        # covers introspection of the physical co_names tuple.
        for a, b in zip(old.co_positions(), new.co_positions()):
            if (move(a[0]), move(a[1]), a[2], a[3]) == b:
                continue
            if not (move(a[0]) == b[0] and a[2] == b[2]
                    and a[1] == first - 1 and b[1] is not None and first <= b[1] < first + count):
                return None
            # A containing def or try span may now end inside the inserted
            # lines. The start of the otherwise identical instruction stayed put.
        constants = None
        for index, (a, b) in enumerate(zip(old.co_consts, new.co_consts)):
            if isinstance(a, CodeType) and isinstance(b, CodeType):
                nested = normalized(a, b)
                if nested is None:
                    return None
                if constants is None:
                    constants = list(new.co_consts)
                constants[index] = nested
        return new.replace(co_consts=tuple(constants) if constants is not None else new.co_consts,
                           co_names=old.co_names,
                           co_firstlineno=old.co_firstlineno, co_linetable=old.co_linetable)

    revised = normalized(before, after)
    return revised is not None and same_code(before, revised)
