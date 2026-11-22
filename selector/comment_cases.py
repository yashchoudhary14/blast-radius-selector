"""Equivalence of parser bodies on the pinned comment-free stress input."""

import ast
import dis
import inspect
from functools import lru_cache
from types import CodeType

from .source_equivalence import _has_dynamic_locals, _same_directives_text_only, same_code
from .state_refs import functions


PARSER = "sqlglot/parser.py"
TOKENIZER = "sqlglot/tokenizer_core.py"


@lru_cache(maxsize=16)
def expression(source):
    return ast.dump(ast.parse(source, mode="eval").body)


def is_expression(node, source):
    return ast.dump(node) == expression(source)


class CommentFree(ast.NodeTransformer):
    """Apply facts about the default parser's built-in tokens and comment lists."""

    def __init__(self, branch, match):
        self.branch = branch
        self.match = match

    def visit_Call(self, node):
        self.generic_visit(node)
        if self.match and is_expression(node.func, "self._match"):
            node.keywords = [
                kw for kw in node.keywords
                if kw.arg != "expression"
                or not is_expression(kw.value, "table if isinstance(table, exp.Expr) else None")
            ]
        return node

    def visit_If(self, node):
        if self.branch and ast.dump(node) == self.branch:
            return []
        return self.generic_visit(node)

    def project(self, body):
        block = ast.Module(body=body, type_ignores=[])
        return self.visit(block).body


def assignment(node, name, value):
    return (isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name
            and is_expression(node.value, value))


def project(tree, path, branches):
    cls_name = "Parser" if path == PARSER else "TokenizerCore"
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or cls.name != cls_name:
            continue
        for fn in cls.body:
            if not isinstance(fn, ast.FunctionDef):
                continue
            if fn.name not in branches and fn.name != "_parse_table_parts":
                continue
            fn.body = CommentFree(branches.get(fn.name), fn.name == "_parse_table_parts").project(fn.body)
            if fn.name == "_add" and not any(isinstance(node, ast.Name) and node.id == "tokens"
                                              and isinstance(node.ctx, ast.Load) for node in ast.walk(fn)):
                fn.body = [node for node in fn.body if not assignment(node, "tokens", "self.tokens")]
    return functions(compile(tree, path, "exec", dont_inherit=True))


def pinned_branches(tree, path):
    # These guard expressions only read built-in fields and locals. Keep the
    # entire pinned statement in the certificate so a change to its bindings
    # or to an intervening expression cannot become invisible.
    specs = {
        "Parser._parse_table_parts_fast": "has_dot and self._prev_comments",
        "Parser._match_text_seq": "len(texts) > 1",
    } if path == PARSER else {
        "TokenizerCore._add": "tokens and (self._comments or tokens[-1].comments)",
    }
    from .analyze import _get_func_node

    guards = {}
    for name, test in specs.items():
        fn = _get_func_node(tree, name)
        for node in ast.walk(fn):
            if isinstance(node, ast.If) and not node.orelse and is_expression(node.test, test):
                guards[fn.name] = ast.dump(node)
    return guards


@lru_cache(maxsize=3)
def pinned_projection(original, path, stress):
    tree = ast.parse(original)
    branches = pinned_branches(tree, path)
    if not stress:
        branches.pop("_match_text_seq", None)
    return project(tree, path, branches)


def equivalent_under_projection(before, projected, after):
    if (before is None or before.co_flags != projected.co_flags or before.co_freevars != projected.co_freevars
            or before.co_cellvars != projected.co_cellvars):
        return False
    removed = set(before.co_varnames) - set(projected.co_varnames)
    if any(op.opname == "LOAD_GLOBAL" and op.argval in removed for op in dis.get_instructions(projected)):
        return False

    def without_locations(code, memo):
        # The stress test checks results and substrings of ParseError, not
        # source positions. Keep constant sharing while dropping locations.
        if id(code) not in memo:
            constants = tuple(without_locations(v, memo) if isinstance(v, CodeType) else v
                              for v in code.co_consts)
            memo[id(code)] = code.replace(co_firstlineno=0, co_linetable=b"", co_consts=constants)
        return memo[id(code)]

    projected, after = without_locations(projected, {}), without_locations(after, {})
    if same_code(projected, after):
        return True
    # These bodies have no nested code or escaping frame. The slot indices,
    # parameter names, exception table and instructions must still agree.
    if any(isinstance(value, CodeType) for value in (*projected.co_consts, *after.co_consts)):
        return False
    argc = (projected.co_argcount + projected.co_kwonlyargcount
            + bool(projected.co_flags & inspect.CO_VARARGS) + bool(projected.co_flags & inspect.CO_VARKEYWORDS))
    if len(projected.co_varnames) != len(after.co_varnames) or projected.co_varnames[:argc] != after.co_varnames[:argc]:
        return False
    return same_code(projected, after.replace(co_varnames=projected.co_varnames))


def unchanged_comment_case(analyzer, edits, candidates, number, revisions, *, stress):
    # The stress test only parses comment-free SELECTs with the default
    # dialect. The TPC-DS executor uses default parsing and DuckDB generation.
    from .analyze import _get_func_node
    from .query_cases import reached_cases

    if not stress:
        if TOKENIZER in edits:
            return False
        # The case traces include imports, optimization and rendering. Both
        # comment mutations must be dormant across these paths.
        tree, _ = analyzer.old_tree_and_lines(PARSER)
        _, masks, files = analyzer.queries.maps[number]
        attach = _get_func_node(tree, "Parser._add_comments").body[0]
        fast = _get_func_node(tree, "Parser._parse_table_parts_fast")
        append = next(node for node in ast.walk(fast) if isinstance(node, ast.If)
                      and is_expression(node.test, "has_dot and self._prev_comments"))
        if reached_cases(masks, files, PARSER,
                         [(branch.body[0].lineno, branch.end_lineno) for branch in (attach, append)]):
            return False

    bit = 1 << number
    for path, source in edits.items():
        if path in candidates.isolated_test_paths:
            if candidates.isolated_test_paths[path] & bit:
                return False
            continue
        if path not in (PARSER, TOKENIZER) or not source or path not in revisions:
            return False
        if analyzer.index["reads"].get(path, 0) & bit:
            return False
        original = (analyzer.base / path).read_bytes()
        if not _same_directives_text_only(original, source):
            return False
        before, scopes = analyzer.old(path)
        new_tree, compiled, after, revised = revisions[path]
        if scopes != revised or before.keys() != after.keys():
            return False
        old_tree, _ = analyzer.old_tree_and_lines(path)
        reached = []
        for name, (_, header, body, _, end) in before.items():
            if header != after[name][1]:
                return False
            if body != after[name][2]:
                old_fn = _get_func_node(old_tree, name)
                locations = analyzer._lines_and_indirect_mask(path, range(old_fn.body[0].lineno, end + 1))
                if not locations & bit:
                    continue
                if _has_dynamic_locals(old_fn) or _has_dynamic_locals(_get_func_node(new_tree, name)):
                    return False
                reached.append(name)
        if not reached:
            continue
        old_normal = pinned_projection(original, path, stress)
        old_code = analyzer.old_code(path)
        new_code = functions(compiled)
        for name in reached:
            normal_old, revised_code = old_normal.get(name), new_code.get(name)
            if (normal_old is None or revised_code is None
                    or not equivalent_under_projection(old_code.get(name), normal_old, revised_code)):
                return False
    return True
