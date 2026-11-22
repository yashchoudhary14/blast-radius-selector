"""Prove that a source edit preserves the compiled program or changes one doc binding."""

import __future__
import ast
import collections.abc
import dis
import inspect
import marshal
import operator
import re
import tokenize
import warnings
from functools import lru_cache
from io import BytesIO
from types import CodeType


DIRECTIVE = re.compile(
    r"#\s*(?:!|.*?coding[=:]|(?:type|fmt|ruff|pyright|mypy|pylint|pragma|isort)\s*:|noqa\b)",
    re.IGNORECASE,
)
_PCT_SPEC = re.compile(r"%(?:\([^)]+\))?[#0\- +]*(?:\*|\d+)?(?:\.(?:\*|\d+))?[hlL]?[diouxXeEfFgGcrsa%]")
_DYN_LOCALS = frozenset({"eval", "exec", "locals", "vars", "dir"})
_LOG_LEVELS = frozenset({"debug", "info", "warning", "warn", "error", "critical", "exception"})


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
        if not isinstance(left, (CodeType, tuple, frozenset)):
            return True
        a, b = id(left), id(right)
        if a in seen or b in reverse:
            return seen.get(a) == b and reverse.get(b) == a
        seen[a], reverse[b] = b, a
        if isinstance(left, CodeType):
            return same_sharing(left.co_consts, right.co_consts)
        return all(same_sharing(x, y) for x, y in zip(left, right))

    return same_sharing(before, after)


@lru_cache(maxsize=64)
def source_directives(source):
    try:
        return tuple((tok.start, tok.string, tok.line) for tok in tokenize.tokenize(BytesIO(source).readline)
                     if tok.type == tokenize.ENCODING or (tok.type == tokenize.COMMENT and DIRECTIVE.match(tok.string)))
    except (SyntaxError, tokenize.TokenError, UnicodeError):
        return None


def same_directives(before, after, insertion=None):
    original, updated = source_directives(before), source_directives(after)
    if original is None or updated is None:
        return False
    first, count = insertion or (0, 0)
    return tuple(((line + count if line >= first else line, col), value)
                 for (line, col), value, _ in original) == tuple((pos, value) for pos, value, _ in updated)


@lru_cache(maxsize=128)
def _compiled_doc(value):
    node = ast.fix_missing_locations(
        ast.Module(body=[ast.Expr(value=ast.Constant(value=value))], type_ignores=[])
    )
    return compile(node, "<doc>", "exec").co_consts[0]


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
    yield "module", "", _doc(tree), True

    def walk(node, prefix="", plain_ancestors=True):
        for stmt in node.body:
            if isinstance(stmt, ast.ClassDef):
                if stmt.decorator_list or getattr(stmt, "type_params", ()):
                    continue
                q = f"{prefix}.{stmt.name}" if prefix else stmt.name
                is_plain = plain_ancestors and _plain_class(stmt)
                yield "class", q, _doc(stmt), is_plain
                walk(stmt, q, is_plain)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if stmt.decorator_list or getattr(stmt, "type_params", ()):
                    continue
                q = f"{prefix}.{stmt.name}" if prefix else stmt.name
                yield "function", q, _doc(stmt), plain_ancestors

    yield from walk(tree)


def _plain_class(node):
    return (isinstance(node, ast.ClassDef)
            and not (node.bases or node.keywords or node.decorator_list or getattr(node, "type_params", ())))


def _doc_assignment(code, value):
    target = _compiled_doc(value)
    instructions = list(dis.get_instructions(code))
    for index, (load, store) in enumerate(zip(instructions, instructions[1:])):
        if (load.opname != "LOAD_CONST" or load.argval != target
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


def _without_doc_value(before, after, original, updated, doc_delta=0, doc_end=0):
    old = _doc_assignment(before, original)
    new = _doc_assignment(after, updated)
    if old is None or old != new:
        return None
    slot, instructions = old
    for index, (left, right) in enumerate(zip(before.co_positions(), after.co_positions())):
        if doc_delta == 0:
            if left != right and (index not in instructions or left[:2] != right[:2]):
                return None
        else:
            if index in instructions:
                if left[0] != right[0] or left[1] + doc_delta != right[1]:
                    return None
            else:
                expected = (
                    left[0] + doc_delta if left[0] is not None and left[0] > doc_end else left[0],
                    left[1] + doc_delta if left[1] is not None and left[1] > doc_end else left[1],
                    left[2],
                    left[3],
                )
                if expected != right:
                    return None
    constants = list(after.co_consts)
    constants[slot] = _compiled_doc(original)
    if doc_delta != 0:
        for i, (a, b) in enumerate(zip(before.co_consts, constants)):
            if isinstance(a, CodeType) and isinstance(b, CodeType):
                nested = _relocate_after(a, b, doc_end + 1, doc_delta)
                if nested is None:
                    return None
                constants[i] = nested
    return after.replace(co_consts=tuple(constants), co_linetable=before.co_linetable)


def _without_fn_doc(before, after, original, updated, doc_delta=0, doc_end=0):
    old_doc = _compiled_doc(original)
    new_doc = _compiled_doc(updated)
    if (not before.co_consts or not after.co_consts
            or before.co_consts[0] != old_doc or after.co_consts[0] != new_doc
            or type(before.co_consts[0]) is not str or type(after.co_consts[0]) is not str):
        return None
    if (any(i.opcode in dis.hasconst and i.arg == 0 for i in dis.get_instructions(before))
            or any(i.opcode in dis.hasconst and i.arg == 0 for i in dis.get_instructions(after))):
        return None
    if doc_delta == 0:
        if list(before.co_positions()) != list(after.co_positions()):
            return None
        return after.replace(co_consts=(old_doc, *after.co_consts[1:]), co_linetable=before.co_linetable)
    for left, right in zip(before.co_positions(), after.co_positions()):
        expected = (
            left[0] + doc_delta if left[0] is not None and left[0] > doc_end else left[0],
            left[1] + doc_delta if left[1] is not None and left[1] > doc_end else left[1],
            left[2],
            left[3],
        )
        if expected != right:
            return None
    constants = [old_doc, *after.co_consts[1:]]
    for i, (a, b) in enumerate(zip(before.co_consts, constants)):
        if i and isinstance(a, CodeType) and isinstance(b, CodeType):
            nested = _relocate_after(a, b, doc_end + 1, doc_delta)
            if nested is None:
                return None
            constants[i] = nested
    return after.replace(co_consts=tuple(constants), co_linetable=before.co_linetable)


def _relocate_after(old, new, first, count):
    def move(line):
        return line + count if line is not None and line >= first else line

    if (old.co_code != new.co_code or move(old.co_firstlineno) != new.co_firstlineno
            or old.co_names != new.co_names or len(old.co_consts) != len(new.co_consts)):
        return None
    for a, b in zip(old.co_positions(), new.co_positions()):
        if (move(a[0]), move(a[1]), a[2], a[3]) != b:
            return None
    constants = list(new.co_consts)
    for index, (a, b) in enumerate(zip(old.co_consts, new.co_consts)):
        if isinstance(a, CodeType) and isinstance(b, CodeType):
            nested = _relocate_after(a, b, first, count)
            if nested is None:
                return None
            constants[index] = nested
    return new.replace(co_consts=tuple(constants), co_firstlineno=old.co_firstlineno, co_linetable=old.co_linetable)


def _replace_nested(parent, target_qname, new_child, doc_delta=0, doc_end=0, before_parent=None):
    if "." not in target_qname:
        matches = [(i, c) for i, c in enumerate(parent.co_consts)
                   if isinstance(c, CodeType) and c.co_qualname == target_qname]
        if len(matches) != 1:
            return None
        idx = matches[0][0]
        if doc_delta == 0 or before_parent is None:
            return parent.replace(co_consts=(*parent.co_consts[:idx], new_child, *parent.co_consts[idx + 1:]))
        # Parent is module or outer class: shift lines after doc_end in parent
        return _relocate_parent_with_child(before_parent, parent, idx, new_child, doc_end + 1, doc_delta)
    outer, _, inner = target_qname.partition(".")
    matches = [(i, c) for i, c in enumerate(parent.co_consts)
               if isinstance(c, CodeType) and c.co_qualname.split(".")[-1] == outer]
    if len(matches) != 1:
        return None
    idx, child = matches[0]
    before_child = before_parent.co_consts[idx] if before_parent and idx < len(before_parent.co_consts) else None
    if not isinstance(before_child, CodeType):
        return None
    updated_child = _replace_nested(child, inner, new_child, doc_delta, doc_end, before_child)
    if updated_child is None:
        return None
    if doc_delta == 0 or before_parent is None:
        return parent.replace(co_consts=(*parent.co_consts[:idx], updated_child, *parent.co_consts[idx + 1:]))
    return _relocate_parent_with_child(before_parent, parent, idx, updated_child, doc_end + 1, doc_delta)


def _relocate_parent_with_child(old, new, child_idx, updated_child, first, count):
    def move(line):
        return line + count if line is not None and line >= first else line

    if (old.co_code != new.co_code or move(old.co_firstlineno) != new.co_firstlineno
            or old.co_names != new.co_names or len(old.co_consts) != len(new.co_consts)):
        return None
    for a, b in zip(old.co_positions(), new.co_positions()):
        expected_end = move(a[1])
        expected_start = move(a[0])
        if (expected_start, expected_end, a[2], a[3]) != b:
            # Containing class/module statement span ends at the shifted end line
            if not (a[0] is not None and a[0] < first and b[0] == a[0] and expected_end == b[1] and a[2] == b[2]):
                return None
    constants = list(new.co_consts)
    for i, (a, b) in enumerate(zip(old.co_consts, new.co_consts)):
        if i == child_idx:
            constants[i] = updated_child
        elif isinstance(a, CodeType) and isinstance(b, CodeType):
            nested = _relocate_after(a, b, first, count)
            if nested is None:
                return None
            constants[i] = nested
    return new.replace(co_consts=tuple(constants), co_firstlineno=old.co_firstlineno, co_linetable=old.co_linetable)


def _find_code(root, qualname):
    matches = [c for c in root.co_consts if isinstance(c, CodeType) and c.co_qualname == qualname]
    if len(matches) == 1:
        return matches[0]
    for c in root.co_consts:
        if isinstance(c, CodeType) and qualname.startswith(c.co_qualname + "."):
            found = _find_code(c, qualname)
            if found is not None:
                return found
    return None


def _class_body(code, name):
    matches = [(i, c) for i, c in enumerate(code.co_consts)
               if isinstance(c, CodeType) and c.co_qualname == name]
    return matches[0] if len(matches) == 1 else (None, None)


def docstring_edit(old_source, source, old_tree, tree, before, after):
    """Accept one existing, doctest-free docstring edit with all live code intact.

    Returns (kind, owner, is_plain, relocated) or None.
    """
    old_docs = {(kind, q): (val, plain) for kind, q, val, plain in _docs(old_tree)}
    for kind, name, revised, is_plain in _docs(tree):
        pair = old_docs.get((kind, name))
        if not pair:
            continue
        original, old_plain = pair
        if original is None or revised is None or original.value == revised.value:
            continue
        if _outside(old_source, original) != _outside(source, revised):
            continue
        delta = (revised.end_lineno - revised.lineno) - (original.end_lineno - original.lineno)
        doc_end = original.end_lineno
        if kind == "module":
            normalized = _without_doc_value(before, after, original.value, revised.value, delta, doc_end)
        elif kind == "class":
            old_body = _find_code(before, name)
            body = _find_code(after, name)
            if old_body is None or body is None:
                return None
            norm_body = _without_doc_value(old_body, body, original.value, revised.value, delta, doc_end)
            if norm_body is None:
                return None
            normalized = _replace_nested(after, name, norm_body, delta, doc_end, before)
        else:
            old_body = _find_code(before, name)
            body = _find_code(after, name)
            if old_body is None or body is None:
                return None
            norm_body = _without_fn_doc(old_body, body, original.value, revised.value, delta, doc_end)
            if norm_body is None:
                return None
            normalized = _replace_nested(after, name, norm_body, delta, doc_end, before)
        if normalized is not None and same_code(before, normalized):
            return kind, name, is_plain and old_plain, delta != 0
        return None
    return None


def docstring_only(old_source, source, old_tree, tree, before, after):
    """Accept one existing, doctest-free docstring with all live code intact."""
    result = docstring_edit(old_source, source, old_tree, tree, before, after)
    return bool(result and result[2] and not result[3])


def redundant_abc_metaclass(old, new):
    originals = {node.name: node for node in old.body if isinstance(node, ast.ClassDef)}
    for cls in new.body:
        if not isinstance(cls, ast.ClassDef) or cls.name not in originals:
            continue
        prior = originals[cls.name]
        if (len(prior.bases) != 1 or ast.unparse(prior.bases[0]) != "abc.ABC"
                or prior.keywords or len(cls.keywords) != 1):
            continue
        keyword = cls.keywords[0]
        if keyword.arg != "metaclass" or ast.unparse(keyword.value) != "abc.ABCMeta":
            continue
        # __build_class__ already resolves this metaclass from the sole base.
        # The explicit keyword is consumed before the same ABCMeta call.
        keywords, cls.keywords = cls.keywords, []
        try:
            if ast.dump(old) == ast.dump(new):
                return True
        finally:
            cls.keywords = keywords
    return False


def _get_func_ast(tree, name):
    node = tree
    for part in name.split(".")[:-1]:
        matches = [n for n in getattr(node, "body", ()) if isinstance(n, ast.ClassDef) and n.name == part]
        if len(matches) != 1:
            return None
        node = matches[0]
    member = name.split(".")[-1]
    matches = [n for n in getattr(node, "body", ())
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == member]
    return matches[0] if len(matches) == 1 else None


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


def postponed_annotation(old_source, source, old_tree, tree, name, before, after):
    if not before.co_flags & after.co_flags & __future__.annotations.compiler_flag:
        return False
    old, revised = _plain_function(old_tree, name), _plain_function(tree, name)
    if old is None or revised is None:
        return False
    old_annotations, _ = _signature_parts(old)
    annotations, _ = _signature_parts(revised)
    return any(_isolated_change(old_source, source, a, b) for a, b in zip(old_annotations, annotations))


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


def _bound_names(node):
    names = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.arg):
            names.add(n.arg)
        elif isinstance(n, ast.alias):
            names.add(n.asname or n.name.partition(".")[0])
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            names.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            names.update(n.names)
        elif hasattr(ast, "MatchAs") and isinstance(n, ast.MatchAs) and n.name:
            names.add(n.name)
        elif hasattr(ast, "MatchStar") and isinstance(n, ast.MatchStar) and n.name:
            names.add(n.name)
        elif hasattr(ast, "MatchMapping") and isinstance(n, ast.MatchMapping) and n.rest:
            names.add(n.rest)
    return names


def _has_dynamic_locals(node):
    return any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in _DYN_LOCALS
               for n in ast.walk(node))


def optional_keyword_only(old_source, source, old_tree, tree, name, before, after):
    """Accept adding unused keyword-only optional parameter(s) with constant defaults.

    Returns (relocated, needs_annotation_eval) or None.
    """
    old = _get_func_ast(old_tree, name)
    revised = _get_func_ast(tree, name)
    if old is None or revised is None:
        return None
    if (old.decorator_list or revised.decorator_list
            or getattr(old, "type_params", ()) or getattr(revised, "type_params", ())):
        return None
    if old.args.kwarg is not None or revised.args.kwarg is not None:
        return None
    if len(revised.args.kwonlyargs) <= len(old.args.kwonlyargs):
        return None
    if len(revised.args.kwonlyargs) != len(revised.args.kw_defaults):
        return None

    old_kw = [(a.arg, ast.dump(a.annotation) if a.annotation else None,
               ast.dump(d) if d is not None else None)
              for a, d in zip(old.args.kwonlyargs, old.args.kw_defaults)]
    new_kw = list(zip(revised.args.kwonlyargs, revised.args.kw_defaults))
    idx = 0
    added = []
    for arg, default in new_kw:
        sig = (arg.arg, ast.dump(arg.annotation) if arg.annotation else None,
               ast.dump(default) if default is not None else None)
        if idx < len(old_kw) and sig == old_kw[idx]:
            idx += 1
        else:
            added.append((arg, default))
    if idx != len(old_kw) or not added:
        return None

    used_names = _bound_names(old)
    if _has_dynamic_locals(old):
        return None
    for arg, default in added:
        if default is None or not isinstance(default, ast.Constant):
            return None
        if arg.arg in used_names:
            return None

    saved_kw, saved_def = revised.args.kwonlyargs, revised.args.kw_defaults
    revised.args.kwonlyargs, revised.args.kw_defaults = old.args.kwonlyargs, old.args.kw_defaults
    same_ast = ast.dump(old_tree) == ast.dump(tree)
    revised.args.kwonlyargs, revised.args.kw_defaults = saved_kw, saved_def
    if not same_ast:
        return None

    delta = (revised.body[0].lineno - old.body[0].lineno) if (old.body and revised.body) else 0
    if delta >= 0:
        if not same_directives(old_source, source, (old.lineno + 1, delta) if delta else None):
            return None
    else:
        if not same_directives(source, old_source, (revised.lineno + 1, -delta)):
            return None

    postponed = bool(before.co_flags & after.co_flags & __future__.annotations.compiler_flag)
    needs_annotation_eval = (not postponed) and any(a.annotation is not None for a, _ in added)
    return delta != 0, needs_annotation_eval


def _same_const(a, b):
    if isinstance(a, CodeType) and isinstance(b, CodeType):
        return same_code(a, b)
    return marshal.dumps(a, 2) == marshal.dumps(b, 2)


def _same_code_except_locals(old, new, forbidden_outer=frozenset()):
    if (old.co_argcount != new.co_argcount or old.co_posonlyargcount != new.co_posonlyargcount
            or old.co_kwonlyargcount != new.co_kwonlyargcount or old.co_flags != new.co_flags
            or old.co_name != new.co_name or old.co_qualname != new.co_qualname
            or old.co_firstlineno != new.co_firstlineno
            or old.co_code != new.co_code or old.co_names != new.co_names
            or old.co_freevars != new.co_freevars or old.co_cellvars != new.co_cellvars
            or old.co_exceptiontable != new.co_exceptiontable
            or len(old.co_consts) != len(new.co_consts)
            or len(old.co_varnames) != len(new.co_varnames)):
        return False
    for a, b in zip(old.co_positions(), new.co_positions()):
        if a[:2] != b[:2]:
            return False

    n_params = (old.co_argcount + old.co_kwonlyargcount
                + bool(old.co_flags & inspect.CO_VARARGS)
                + bool(old.co_flags & inspect.CO_VARKEYWORDS))
    if old.co_varnames[:n_params] != new.co_varnames[:n_params]:
        return False

    blocked = set(old.co_names) | set(old.co_freevars) | set(old.co_cellvars)
    renames = {}
    for ov, nv in zip(old.co_varnames[n_params:], new.co_varnames[n_params:]):
        if ov != nv:
            if ov in blocked or nv in blocked or nv in old.co_varnames or ov in new.co_varnames:
                return False
            if ov in renames and renames[ov] != nv:
                return False
            renames[ov] = nv
    if len(set(renames.values())) != len(renames):
        return False

    local_names = set(renames) | set(renames.values()) | forbidden_outer
    if forbidden_outer & (set(old.co_names) | set(old.co_freevars) | set(old.co_cellvars) | set(old.co_varnames)
                          | set(new.co_names) | set(new.co_freevars) | set(new.co_cellvars) | set(new.co_varnames)):
        return False

    for a, b in zip(old.co_consts, new.co_consts):
        if isinstance(a, CodeType) and isinstance(b, CodeType):
            if not _same_code_except_locals(a, b, local_names):
                return False
        elif not _same_const(a, b):
            return False
    return True


def local_rename_only(old_source, source, old_tree, tree, old_codes, new_codes, changed_funcs):
    """Accept pure non-parameter local variable renames inside function bodies."""
    if not changed_funcs or not same_directives(old_source, source):
        return False
    for q in changed_funcs:
        old_c, new_c = old_codes.get(q), new_codes.get(q)
        old_fn = _get_func_ast(old_tree, q)
        if old_c is None or new_c is None or old_fn is None:
            return False
        if _has_dynamic_locals(old_fn):
            return False
        if not _same_code_except_locals(old_c, new_c):
            return False
    return True


def _same_msg_expr(a, b, *, allow_empty_to_fstring=True):
    if a is None and b is None:
        return True
    if a is None:
        return isinstance(b, ast.Constant) and type(b.value) is str
    if b is None:
        return isinstance(a, ast.Constant) and type(a.value) is str

    def fvals(node):
        if isinstance(node, ast.Constant) and type(node.value) is str:
            return []
        if isinstance(node, ast.JoinedStr):
            vals = []
            for v in node.values:
                if isinstance(v, ast.Constant) and type(v.value) is str:
                    continue
                if isinstance(v, ast.FormattedValue):
                    vals.append(ast.dump(v))
                else:
                    return None
            return vals
        return None

    fa, fb = fvals(a), fvals(b)
    if fa is None or fb is None or fa != fb:
        return False
    if not allow_empty_to_fstring and a is None and fb:
        return False
    return True


def _collect_ast_diffs(a, b, diffs):
    if type(a) is not type(b):
        diffs.append((a, b))
        return
    if isinstance(a, list):
        if len(a) != len(b):
            diffs.append((a, b))
            return
        for x, y in zip(a, b):
            _collect_ast_diffs(x, y, diffs)
        return
    if not isinstance(a, ast.AST):
        if a != b:
            diffs.append((a, b))
        return
    if isinstance(a, ast.Constant):
        if type(a.value) is not type(b.value) or a.value != b.value:
            diffs.append((a, b))
        return
    if isinstance(a, ast.Assert) or _log_call_match(a, b) is not None:
        if ast.dump(a) != ast.dump(b):
            diffs.append((a, b))
        return
    for field in a._fields:
        va, vb = getattr(a, field), getattr(b, field)
        if type(va) is not type(vb) or (isinstance(va, list) and len(va) != len(vb)):
            diffs.append((a, b))
            return
    sub = []
    for field in a._fields:
        _collect_ast_diffs(getattr(a, field), getattr(b, field), sub)
    if any(not isinstance(x, ast.AST) or not isinstance(y, ast.AST) for x, y in sub):
        diffs.append((a, b))
    else:
        diffs.extend(sub)


def _pct_specs(s):
    return [m for m in _PCT_SPEC.findall(s) if m != "%%"]


def _log_call_match(a, b):
    if not (isinstance(a, ast.Call) and isinstance(b, ast.Call)):
        return None
    if ast.dump(a.func) != ast.dump(b.func):
        return None
    fn = a.func
    if not (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name)
            and fn.value.id in ("logger", "logging") and fn.attr in _LOG_LEVELS):
        return None
    # Check message is either args[0] (with identical remaining args and keywords) or kw 'msg'
    msg_a = msg_b = None
    has_extra_args = False
    if a.args and b.args and len(a.args) == len(b.args) and ast.dump(ast.Tuple(elts=a.args[1:])) == ast.dump(ast.Tuple(elts=b.args[1:])):
        if len(a.keywords) == len(b.keywords) and all(ast.dump(ka) == ast.dump(kb) for ka, kb in zip(a.keywords, b.keywords)):
            msg_a, msg_b = a.args[0], b.args[0]
            has_extra_args = len(a.args) > 1
    if msg_a is None and len(a.args) == len(b.args) == 0 and len(a.keywords) == len(b.keywords):
        ka_map = {k.arg: k.value for k in a.keywords}
        kb_map = {k.arg: k.value for k in b.keywords}
        if ka_map.keys() == kb_map.keys() and "msg" in ka_map:
            if all(k == "msg" or ast.dump(ka_map[k]) == ast.dump(kb_map[k]) for k in ka_map):
                msg_a, msg_b = ka_map["msg"], kb_map["msg"]
    if msg_a is None or not _same_msg_expr(msg_a, msg_b):
        return None
    if has_extra_args:
        if not (isinstance(msg_a, ast.Constant) and type(msg_a.value) is str
                and isinstance(msg_b, ast.Constant) and type(msg_b.value) is str
                and _pct_specs(msg_a.value) == _pct_specs(msg_b.value)):
            return None
    return fn.attr


def text_message_only(old_source, source, old_tree, tree, changed_funcs):
    """Accept edits that only alter logger message text or assert failure text.

    Returns (kind, lines, relocated, include_stderr) where kind is 'log' or 'assert', or None.
    """
    if not changed_funcs:
        return None
    diffs = []
    for q in changed_funcs:
        old_fn = _get_func_ast(old_tree, q)
        new_fn = _get_func_ast(tree, q)
        if old_fn is None or new_fn is None:
            return None
        _collect_ast_diffs(old_fn, new_fn, diffs)
    if not diffs:
        return None

    # Check if all diffs are logger.<level>(msg, ...) or ast.Assert(test, msg)
    kinds = set()
    lines = set()
    relocated = False
    include_stderr = False
    for a, b in diffs:
        if isinstance(a, ast.Assert) and isinstance(b, ast.Assert):
            if ast.dump(a.test) != ast.dump(b.test) or not _same_msg_expr(a.msg, b.msg, allow_empty_to_fstring=False):
                return None
            kinds.add("assert")
            lines.update(range(a.lineno, (a.end_lineno or a.lineno) + 1))
            if (a.end_lineno or a.lineno) - a.lineno != (b.end_lineno or b.lineno) - b.lineno:
                relocated = True
        else:
            lvl = _log_call_match(a, b)
            if lvl is None:
                return None
            kinds.add("log")
            if lvl not in ("debug", "info"):
                include_stderr = True
            lines.update(range(a.lineno, (a.end_lineno or a.lineno) + 1))
            if (a.end_lineno or a.lineno) - a.lineno != (b.end_lineno or b.lineno) - b.lineno:
                relocated = True
    if len(kinds) != 1:
        return None
    if not relocated:
        if not same_directives(old_source, source):
            return None
    else:
        # Verify directives modulo line shifts if any
        if not _same_directives_text_only(old_source, source):
            return None
    return next(iter(kinds)), lines, relocated, include_stderr


def _same_directives_text_only(before, after):
    original, updated = source_directives(before), source_directives(after)
    return (original is not None and updated is not None
            and tuple(value for _, value, _ in original) == tuple(value for _, value, _ in updated))


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


def _literal_binding(node):
    if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
        return None
    if all(isinstance(n, (ast.Constant, ast.Tuple, ast.Load)) for n in ast.walk(node.value)):
        return node.targets[0].id
    return None


def _static_import(node):
    if not isinstance(node, ast.ImportFrom) or node.level:
        return False
    # These stdlib modules publish concrete bindings, without lazy attribute lookup.
    module = {"collections.abc": collections.abc, "operator": operator}.get(node.module)
    return module is not None and all(a.name in vars(module) and not a.name.startswith("_") for a in node.names)


def _dialect_classes(tree, path):
    if not path.startswith("sqlglot/dialects/") or not any(
        isinstance(n, ast.ImportFrom) and n.level == 0 and n.module == "sqlglot.dialects.dialect"
        and any(a.name == "Dialect" and a.asname is None for a in n.names) for n in tree.body
    ):
        return []
    # The pinned Dialect metaclass uses an ordinary namespace, reads its
    # members by key and has no order-dependent hook for plain scalar data.
    return [n for n in tree.body if isinstance(n, ast.ClassDef) and not n.keywords and not n.decorator_list
            and len(n.bases) == 1 and isinstance(n.bases[0], ast.Name) and n.bases[0].id == "Dialect"]


def _binds(node, name):
    if isinstance(node, ast.Name):
        return node.id == name and not isinstance(node.ctx, ast.Load)
    if isinstance(node, ast.alias):
        return (node.asname or node.name.partition(".")[0]) in (name, "*")
    if isinstance(node, ast.arg):
        return node.arg == name
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name in (name, "__getattr__")
    return False


def type_checking_imports(old, new):
    """Verify only imports in existing, inactive module-level typing guards changed."""
    if len(old.body) != len(new.body):
        return False

    def imports(stmt):
        return (isinstance(stmt, ast.ImportFrom) and stmt.module != "__future__"
                and all(alias.name != "*" for alias in stmt.names))

    bindings = {}
    restored = []
    for before, after in zip(old.body, new.body):
        if isinstance(before, ast.Import):
            for alias in before.names:
                if alias.name == "typing":
                    bindings[alias.asname or alias.name] = alias
        if not isinstance(before, ast.If) or not isinstance(after, ast.If):
            continue
        test = before.test
        if not (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
                and isinstance(test.value, ast.Name) and test.value.id in bindings):
            continue

        if [ast.dump(stmt) for stmt in before.body] == [ast.dump(stmt) for stmt in after.body]:
            continue
        if ([ast.dump(stmt) for stmt in before.body if not imports(stmt)]
                != [ast.dump(stmt) for stmt in after.body if not imports(stmt)]):
            return False
        name = test.value.id
        # Comprehension iteration variables cannot rebind the module name.
        local_targets = {target for node in ast.walk(old) if isinstance(node, ast.comprehension)
                         for target in ast.walk(node.target)}
        if any((node is not bindings[name] and node not in local_targets and _binds(node, name))
               or isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name == name
               or isinstance(node, ast.MatchMapping) and node.rest == name
               or (isinstance(node, ast.Attribute) and node.attr == "TYPE_CHECKING"
                   and not isinstance(node.ctx, ast.Load)) for node in ast.walk(old)):
            return False
        restored.append((after, after.body, before.body))

    for node, _, original in restored:
        node.body = original
    try:
        return bool(restored) and ast.dump(old) == ast.dump(new)
    finally:
        for node, updated, _ in restored:
            node.body = updated


def guarded_type_aliases(old, new, before, after):
    """Recognize postponed annotation edits and a new, inactive type-alias block."""
    from .core_headers import inert_header

    possible = (ast.Name, ast.Attribute, ast.Subscript, ast.Tuple, ast.List, ast.BinOp,
                ast.BitOr, ast.Constant, ast.Load)
    original_guards = {ast.dump(node) for node in old.body if isinstance(node, ast.If)}
    aliases = {}
    kept, found = [], False
    for stmt in new.body:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                if alias.name == "typing":
                    aliases[alias.asname or alias.name] = alias
        test = stmt.test if isinstance(stmt, ast.If) else None
        if not (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
                and isinstance(test.value, ast.Name) and test.value.id in aliases
                and ast.dump(stmt) not in original_guards):
            kept.append(stmt)
            continue
        name = test.value.id
        if (found or stmt.orelse or any(
                not (isinstance(node, ast.Assign) and len(node.targets) == 1
                     and isinstance(node.targets[0], ast.Name) and not node.targets[0].id.startswith("__")
                     and node.targets[0].id != name
                     and all(isinstance(child, possible) for child in ast.walk(node.value)))
                for node in stmt.body)):
            return False
        if any((node is not aliases[name] and _binds(node, name))
               or isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name == name
               or isinstance(node, ast.MatchMapping) and node.rest == name
               or isinstance(node, ast.Attribute) and node.attr == "TYPE_CHECKING"
                  and not isinstance(node.ctx, ast.Load)
               for node in ast.walk(new)):
            return False
        found = True
    if not found or len(old.body) != len(kept):
        return False
    for a, b in zip(old.body, kept):
        if ast.dump(a) == ast.dump(b):
            continue
        if (not isinstance(a, ast.FunctionDef) or not isinstance(b, ast.FunctionDef)
                or inert_header(a, b, before.co_flags, after.co_flags) != "annotation"):
            return False
        args, returns = b.args, b.returns
        b.args, b.returns = a.args, a.returns
        try:
            if ast.dump(a) != ast.dump(b):
                return False
        finally:
            b.args, b.returns = args, returns
    return True


def added_binding(old, new, insertion, path, compiled):
    """Find one fresh binding with no eager computation besides a cached import."""
    first, count = insertion
    for scope in [new, *_dialect_classes(new, path)]:
        for index, stmt in enumerate(scope.body):
            if not (first <= stmt.lineno <= stmt.end_lineno < first + count):
                continue
            name = _literal_binding(stmt)
            if scope is new and isinstance(stmt, ast.FunctionDef):
                args = stmt.args
                annotations, _ = _signature_parts(stmt)
                if ((not any(annotations) or compiled.co_flags & __future__.annotations.compiler_flag)
                        and not stmt.decorator_list and not getattr(stmt, "type_params", ())
                        and not args.defaults and not any(args.kw_defaults)):
                    name = stmt.name
            if scope is new and _static_import(stmt) and len(stmt.names) == 1 and stmt.module == "operator":
                # collections imports operator before returning, so the added
                # statement only retrieves an existing module and attribute.
                if any(isinstance(n, ast.ImportFrom) and not n.level and n.module == "collections"
                       for n in scope.body[:index]):
                    alias = stmt.names[0]
                    name = alias.asname or alias.name
            if (not name or name.startswith("__") or name.endswith("_sql")
                    or not isinstance(stmt, ast.ImportFrom) and not name.startswith("_")):
                continue
            if any(_binds(n, name) or (isinstance(n, ast.Name) and n.id == "__all__") for n in ast.walk(old)):
                continue
            body = scope.body
            scope.body = body[:index] + body[index + 1:]
            try:
                if not index and ast.get_docstring(scope, clean=False) is not None:
                    continue
                if ast.dump(old) == ast.dump(new):
                    return name, scope is new
            finally:
                scope.body = body
    return None


def _forwarded_expression(node, argument):
    """Match two expression forms that read only a forwarded argument."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        call = node.operand
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "bool"
                and not call.keywords and len(call.args) == 1 and isinstance(call.args[0], ast.Name)
                and call.args[0].id == argument):
            return "truth"
    if isinstance(node, ast.GeneratorExp) and len(node.generators) == 1:
        loop = node.generators[0]
        if (not loop.is_async and len(loop.ifs) == 1 and isinstance(loop.iter, ast.Name) and loop.iter.id == argument
                and all(isinstance(n, ast.Name) for n in (loop.target, node.elt, loop.ifs[0]))
                and loop.target.id == node.elt.id == loop.ifs[0].id):
            return "filter"
    return None


def extracted_expression(old_source, source, old, new, name, path, compiled):
    """Check an isolated extraction, leaving call-frame observers to the selector."""
    helper = _plain_function(new, name)
    if not name.startswith("_") or helper is None or len(helper.body) != 1 or not isinstance(helper.body[0], ast.Return):
        return False
    args = helper.args
    parameters = args.posonlyargs + args.args
    if len(parameters) != 1 or args.vararg or args.kwarg or args.kwonlyargs:
        return False
    form = _forwarded_expression(helper.body[0].value, parameters[0].arg)
    if form is None or (form == "truth" and parameters[0].arg == "bool"):
        return False
    for fn in new.body:
        if not isinstance(fn, ast.FunctionDef) or fn is helper or not _plain_function(new, fn.name):
            continue
        original = _plain_function(old, fn.name)
        if original is None:
            continue
        for stmt in fn.body:
            for call in ast.walk(stmt):
                if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != name
                        or call.keywords or len(call.args) != 1 or not isinstance(call.args[0], ast.Name)):
                    continue
                argument = call.args[0].id
                if form == "filter":
                    # The iterable is the caller's unchanged varargs tuple.
                    if original.args.vararg is None or original.args.vararg.arg != argument or any(
                        _binds(n, argument) for n in ast.walk(original) if n is not original.args.vararg
                    ):
                        continue
                elif any(_binds(n, "bool") for n in ast.walk(original)):
                    continue
                for expr in ast.walk(original):
                    if _forwarded_expression(expr, argument) != form:
                        continue
                    prefix, suffix = _outside(old_source, expr)
                    left, right = _outside(source, call)
                    body = old_source[len(prefix):len(old_source) - len(suffix)]
                    # A sole generator argument can share its parentheses
                    # with the containing call. Try restoring its inner text.
                    choices = (body, body[1:-1]) if form == "filter" and body[:1] == b"(" and body[-1:] == b")" else (body,)
                    for value in choices:
                        restored = left + value + right
                        insertion = line_insertion(old_source, restored)
                        if insertion is None:
                            continue
                        try:
                            restored_tree = ast.parse(restored, path)
                        except (SyntaxError, ValueError):
                            continue
                        if (added_binding(old, restored_tree, insertion, path, compiled) == (name, True)
                                and same_directives(old_source, source, insertion)):
                            return True
    return False


def reordered_statements(old_source, source, old, new, path):
    """Recognize a permutation of independent imports or class assignments."""
    def imports(node):
        if not _static_import(node) or len(node.names) < 2:
            return None
        if len({a.asname or a.name for a in node.names}) == len(node.names):
            return node.module, sorted((a.name, a.asname or a.name) for a in node.names)
        return None

    def pairs(a, b):
        return (a.lineno, a.end_lineno) == (b.lineno, b.end_lineno)

    for a, b in zip(old.body, new.body):
        if isinstance(a, ast.ImportFrom) and isinstance(b, ast.ImportFrom) and pairs(a, b):
            order = imports(a)
            if order is not None and order == imports(b) and _isolated_change(old_source, source, a, b):
                return True
    for a, b in zip(_dialect_classes(old, path), _dialect_classes(new, path)):
        if a.name != b.name:
            continue
        for i in range(min(len(a.body), len(b.body)) - 1):
            left, right = a.body[i:i + 2]
            first, last = b.body[i:i + 2]
            l_name, r_name = _literal_binding(left), _literal_binding(right)
            if (l_name and r_name and l_name != r_name and not l_name.startswith("__")
                    and not r_name.startswith("__") and pairs(left, first) and pairs(right, last)
                    and ast.dump(left) == ast.dump(last) and ast.dump(right) == ast.dump(first)):
                # Compare the exact bytes before and after the swapped pair.
                if (_outside(old_source, left)[0] == _outside(source, first)[0]
                        and _outside(old_source, right)[1] == _outside(source, last)[1]
                        and right.lineno == left.end_lineno + 1 and last.lineno == first.end_lineno + 1):
                    return True
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
