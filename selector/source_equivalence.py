"""Prove that a source edit preserves the compiled program or changes one doc binding."""

import __future__
import ast
import dis
import inspect
import marshal
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
    def directives(source):
        return [(tok.start[0], tok.string) for tok in tokenize.tokenize(BytesIO(source).readline)
                if tok.type == tokenize.ENCODING or (tok.type == tokenize.COMMENT and DIRECTIVE.match(tok.string))]

    try:
        b_dirs = [s for _, s in directives(before)]
        a_dirs = [s for _, s in directives(after)]
        return b_dirs == a_dirs
    except (SyntaxError, tokenize.TokenError, UnicodeError):
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
