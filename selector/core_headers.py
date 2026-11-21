"""Classify shared function headers whose evaluation has no user code."""

import __future__
import ast

from .config_edits import _value, added_methods
from .scopes import snapshot


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
            if not all(isinstance(value, ast.Constant) and type(value.value) in (type(None), bool, int, str)
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


def private_constants(before, after):
    """Project private, inert module bindings out of otherwise unchanged headers."""
    types = {"int", "str", "bool", "float"}

    def atomic_type(node):
        if isinstance(node, ast.Name):
            return node.id in types
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "type"
                and len(node.args) == 1 and isinstance(node.args[0], ast.Constant)
                and node.args[0].value is None and not node.keywords)

    def declarations(tree):
        found = {}
        for node in tree.body:
            if (not isinstance(node, ast.AnnAssign) or not node.simple
                    or not isinstance(node.target, ast.Name) or not node.value):
                continue
            name = node.target.id
            if not name.startswith("_") or not name[1:].isupper():
                continue
            value = node.value
            # Both builtins are resolved by the unchanged module preamble.
            if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                    and value.func.id == "frozenset" and not value.keywords
                    and len(value.args) == 1 and isinstance(value.args[0], ast.Tuple)
                    and all(atomic_type(item) for item in value.args[0].elts)):
                found[name] = node
            elif _value(value):
                found[name] = node
        return found

    old, new = ast.parse(before), ast.parse(after)
    left, right = declarations(old), declarations(new)
    names = left.keys() - right.keys()
    if not names:
        return set()
    old.body = [node for node in old.body if node not in {left[n] for n in names}]
    protected = names | types | {"frozenset", "type"}
    for tree in (old, new):
        if any(isinstance(node, ast.Name) and node.id in protected and not isinstance(node.ctx, ast.Load)
               for node in ast.walk(tree)):
            return set()
        if any(isinstance(node, ast.alias) and (node.asname or node.name) in protected | {"*"}
               for node in ast.walk(tree)):
            return set()
    old_functions, old_scopes = snapshot(old, before)
    new_functions, new_scopes = snapshot(new, after)
    if (old_scopes == new_scopes and old_functions.keys() == new_functions.keys()
            and all(old_functions[n][1] == new_functions[n][1] for n in old_functions)):
        return names
    return set()


def schema_methods(before, after, before_funcs, after_funcs):
    """Recognize ordinary method binding changes on the pinned schema classes."""
    # AbstractMappingSchema has no subclass hook and Schema uses standard
    # ABCMeta. No registration callback discovers the plain methods here.
    allowed = {"AbstractMappingSchema", "MappingSchema"}
    removed = before_funcs.keys() - after_funcs.keys()
    created = after_funcs.keys() - before_funcs.keys()
    if not (removed or created) or any(q.rpartition(".")[0] not in allowed for q in removed | created):
        return set()
    if any(before_funcs[q][1] != after_funcs[q][1] for q in before_funcs.keys() & after_funcs.keys()):
        return set()

    def project(source, omitted):
        tree = ast.parse(source)
        for cls in tree.body:
            if isinstance(cls, ast.ClassDef) and cls.name in allowed:
                cls.body = [node for node in cls.body
                            if not isinstance(node, ast.FunctionDef)
                            or f"{cls.name}.{node.name}" not in omitted]
        return tree

    old, new = ast.parse(before), ast.parse(after)
    old_shell, new_shell = project(before, removed), project(after, created)
    gone = added_methods(old_shell, old, allow_body_edits=True) if removed else set()
    added = added_methods(new_shell, new, allow_body_edits=True) if created else set()
    if gone == {q.rsplit(".", 1)[1] for q in removed} and added == {q.rsplit(".", 1)[1] for q in created}:
        return gone | added
    return set()


def retired_dialect_bindings(before, after):
    """Project inert declarations from the shared Dialect registry."""
    def simple_enum(node):
        if (not isinstance(node, ast.ClassDef) or node.decorator_list or node.keywords
                or [base.id if isinstance(base, ast.Name) else None for base in node.bases] != ["str", "AutoName"]):
            return False
        members = [stmt for stmt in node.body if not string_statement(stmt)]
        return bool(members) and all(
            isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id.isupper()
            and not stmt.targets[0].id.startswith("_") and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == "auto"
            and not stmt.value.args and not stmt.value.keywords for stmt in members
        )

    def string_statement(node):
        return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and type(node.value.value) is str

    def declarations(cls):
        result = {}
        for i, node in enumerate(cls.body):
            if isinstance(node, ast.FunctionDef) and not node.decorator_list and not node.name.startswith("_"):
                result[node.name] = (node,)
                continue
            target = node.target if isinstance(node, ast.AnnAssign) and node.simple else None
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
            if (not isinstance(target, ast.Name) or not target.id.isupper() or target.id.startswith("_")
                    or not node.value or not (_value(node.value) or isinstance(node.value, ast.Dict)
                                              and not node.value.keys)):
                continue
            following = cls.body[i + 1:i + 2]
            result[target.id] = (node, *following) if following and string_statement(following[0]) else (node,)
        return result

    old, new = ast.parse(before), ast.parse(after)
    new_names = {node.name for node in new.body if isinstance(node, ast.ClassDef)}
    enums = {node for node in old.body if simple_enum(node) and node.name not in new_names}
    if not enums:
        return {}
    classes = [next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Dialect"), None)
               for tree in (old, new)]
    if None in classes:
        return {}
    left, right = (declarations(cls) for cls in classes)
    changed = left.keys() ^ right.keys()
    if not changed or any(
            isinstance(node, ast.FunctionDef) and (
                name in right or getattr(node, "type_params", ())
                or not all(value is None or _value(value) for value in (*node.args.defaults, *node.args.kw_defaults))
            )
            for name in changed for node in (*left.get(name, ()), *right.get(name, ()))
    ):
        return {}
    old.body = [node for node in old.body if node not in enums]
    for cls, bindings in zip(classes, (left, right), strict=True):
        removed = {node for name in changed for node in bindings.get(name, ())}
        cls.body = [node for node in cls.body if node not in removed]
    if ast.dump(old) != ast.dump(new):
        return {}
    names = changed | {node.name for node in enums}
    # The metaclass must not consume these names while constructing dialects.
    meta = next((node for node in old.body if isinstance(node, ast.ClassDef) and node.name == "_Dialect"), None)
    if meta is None or any((isinstance(node, ast.Name) and node.id in names)
           or (isinstance(node, ast.Attribute) and node.attr in names)
           or (isinstance(node, ast.Constant) and type(node.value) is str and node.value in names)
           for node in ast.walk(meta)):
        return {}
    return {name: left.get(name, ()) for name in names}
