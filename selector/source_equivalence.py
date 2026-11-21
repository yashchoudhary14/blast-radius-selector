"""Prove that a source edit preserves the compiled program or changes one doc binding."""

import __future__
import ast
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


def _undecorated_class(node):
    return (isinstance(node, ast.ClassDef) and not (
        node.keywords or node.decorator_list or getattr(node, "type_params", ())))


def _plain_class(node):
    return _undecorated_class(node) and not node.bases


def docstring_only(old_source, source, old_tree, tree):
    """Locate an isolated doc value change with stable source line numbers."""
    scopes = [(old_tree, tree)]
    scopes.extend((a, b) for a, b in zip(old_tree.body, tree.body)
                  if _undecorated_class(a) and _undecorated_class(b) and a.name == b.name)
    for before, after in scopes:
        if _isolated_change(old_source, source, _doc(before), _doc(after)):
            return before
    return None


def doc_class_safe(node, tree, path, load_tree):
    """Admit ordinary classes and audited base classes with doc-independent construction."""
    def bindings(name, preceding):
        result = []
        for index, top in enumerate(preceding):
            for stmt in ast.walk(top):
                if isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == name:
                    result.append((index, stmt))
                elif isinstance(stmt, (ast.Import, ast.ImportFrom)) and any(
                    (a.asname or a.name.partition(".")[0]) in (name, "*") for a in stmt.names
                ):
                    result.append((index, stmt))
                elif isinstance(stmt, ast.Name) and stmt.id == name and not isinstance(stmt.ctx, ast.Load):
                    result.append((index, stmt))
        return result

    active = set()

    def safe(cls, body, source):
        if cls in active:
            return False
        if source == "sqlglot/expressions/core.py" and cls.name == "Expr" and cls is not node:
            # The pinned trait is an identity decorator. Expr's only doc use
            # at subclass creation copies the received str back to __doc__.
            return True
        if not _undecorated_class(cls) or (cls is not node and any(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "__init_subclass__"
            for n in cls.body
        )):
            return False
        preceding = body[:body.index(cls)]
        active.add(cls)
        for base in cls.bases:
            if not safe_base(base, preceding, source):
                return False
        active.remove(cls)
        return True

    def safe_base(base, preceding, source):
        name = base.value if isinstance(base, ast.Attribute) and base.attr == "ABC" else base
        if not isinstance(name, ast.Name):
            return False
        matches = bindings(name.id, preceding)
        if len(matches) != 1:
            return False
        index, statement = matches[0]
        if statement is not preceding[index]:
            return False
        if name is not base:
            return isinstance(statement, ast.Import) and any(
                a.name == "abc" and (a.asname or a.name) == name.id for a in statement.names)
        if isinstance(statement, ast.ClassDef):
            return safe(statement, preceding, source)
        if not isinstance(statement, ast.ImportFrom) or statement.level or not statement.module:
            return False
        imported = [a.name for a in statement.names if (a.asname or a.name) == name.id]
        if len(imported) != 1:
            return False
        # EnumType treats __doc__ as a special attribute. ABCMeta only
        # checks member values for __isabstractmethod__, which str lacks.
        if (statement.module, imported[0]) in {("enum", "Enum"), ("abc", "ABC")}:
            return True
        if not statement.module.startswith("sqlglot.expressions."):
            return False
        target = statement.module.replace(".", "/") + ".py"
        other = load_tree(target)
        classes = bindings(imported[0], other.body)
        return (len(classes) == 1 and isinstance(classes[0][1], ast.ClassDef)
                and other.body[classes[0][0]] is classes[0][1] and safe(classes[0][1], other.body, target))

    return safe(node, tree.body, path)


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
                # An assignment to a simple name in a literal-false branch
                # introduces no class or module binding. Inside a function it
                # could make a name local even though it is never assigned.
                dead_branch = (isinstance(owner, (ast.Module, ast.ClassDef)) and field == "body"
                               and index > 0 and isinstance(stmt, ast.If) and not stmt.orelse
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
                if dead_branch or dead_raise:
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
