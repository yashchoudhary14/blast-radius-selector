"""Classify shared function headers whose evaluation has no user code."""

import __future__
import ast


CORE_FILES = frozenset({
    "sqlglot/parser.py", "sqlglot/generator.py", "sqlglot/tokens.py",
    "sqlglot/tokenizer_core.py", "sqlglot/dialects/dialect.py", "sqlglot/helper.py",
    "sqlglot/schema.py", "sqlglot/transforms.py",
})


def shared_core(path):
    return path in CORE_FILES or path.startswith(("sqlglot/expressions/", "sqlglot/optimizer/"))


def _arguments(fn):
    args = fn.args
    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + args.defaults
    return (
        tuple(zip((a.arg for a in args.posonlyargs), defaults)),
        tuple(zip((a.arg for a in args.args), defaults[len(args.posonlyargs):])),
        tuple(zip((a.arg for a in args.kwonlyargs), args.kw_defaults)),
        args.vararg.arg if args.vararg else None,
        args.kwarg.arg if args.kwarg else None,
    )


def inert_header(before, after, original_flags, updated_flags):
    """Return metadata-only, optional positional change, or unknown."""
    if (type(before) is not type(after) or before is None or before.decorator_list
            or after.decorator_list or getattr(before, "type_params", ())
            or getattr(after, "type_params", ()) or before.type_comment != after.type_comment
            or before.body[0].lineno == before.lineno or after.body[0].lineno == after.lineno):
        return None
    if not original_flags & updated_flags & __future__.annotations.compiler_flag:
        return None
    left, right = _arguments(before), _arguments(after)

    def dump(group):
        return tuple((name, ast.dump(default) if default is not None else None) for name, default in group)

    if left[3:] != right[3:] or dump(left[0]) != dump(right[0]):
        return None

    def literal_defaults(group, updated):
        if tuple(name for name, _ in group) != tuple(name for name, _ in updated):
            return False
        for (_, a), (_, b) in zip(group, updated):
            if a is b or (a is not None and b is not None and ast.dump(a) == ast.dump(b)):
                continue
            if not all(isinstance(value, ast.Constant) and type(value.value) in (type(None), bool)
                       for value in (a, b)):
                return False
        return True

    if any(dump(left[i]) != dump(right[i]) for i in (1, 2)) and all(
        literal_defaults(left[i], right[i]) for i in (1, 2)
    ):
        return "default"

    def suffix(a, b):
        shorter, longer = sorted((a, b), key=len)
        if dump(shorter) != dump(longer[:len(shorter)]):
            return None
        tail = longer[len(shorter):]
        if any(not isinstance(default, ast.Constant) or type(default.value) not in (type(None), bool)
               for _, default in tail):
            return None
        return len(b) - len(a)

    positional = suffix(left[1], right[1])
    keyword = suffix(left[2], right[2])
    if positional is None or keyword is None or (positional and keyword):
        return None
    if not positional and not keyword:
        return "annotation"
    if left[4] is not None or (positional and left[3] is not None):
        return None
    return "add" if positional > 0 or keyword > 0 else "remove"


def only_nested_definitions(before, after, inert):
    """Check a closure factory whose only edits are to deferred child code."""
    if before is None or after is None:
        return False

    def shell(fn):
        names = [stmt.name for stmt in fn.body if isinstance(stmt, ast.FunctionDef)]
        return [stmt.name if isinstance(stmt, ast.FunctionDef) and stmt.name in inert
                and names.count(stmt.name) == 1 else ast.dump(stmt)
                for stmt in fn.body]

    return shell(before) == shell(after)
