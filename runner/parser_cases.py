"""Run the original pinned parser assertions in independently selected groups."""

import ast
import hashlib
import inspect
import textwrap
from types import FunctionType, MethodType


MAX_NODES = "tests/test_parser.py::TestParser::test_max_nodes"
GROUPS = (tuple(range(4)), tuple(range(4, 11)), tuple(range(11, 13)), tuple(range(13, 15)))
METHOD_SOURCE = "83e19d65f6ac96ead3dd16737c5def4daf304786ce514f31fa69491d5aecb8e7"


def parser_method(method):
    first_line = method.__code__.co_firstlineno
    source = textwrap.dedent(inspect.getsource(method))
    if hashlib.sha256(source.encode()).hexdigest() != METHOD_SOURCE:
        raise ValueError("parser case filter requires the pinned test method")
    return ast.increment_lineno(ast.parse(source), first_line - 1)


def filter_parser(case, indices):
    wanted = set(indices)
    if not wanted or any(type(i) is not int or not 0 <= i < len(GROUPS) for i in wanted):
        raise ValueError("unknown parser case")
    method = case.test_max_nodes
    tree = parser_method(method)
    # Both large assertions use the same SQL string. Keep its construction
    # when the second assertion runs alone. Keep the reset check paired.
    statements = {j for i in wanted for j in GROUPS[i]}
    if 2 in wanted:
        statements.update(range(4, 9))
    tree.body[0].body = [stmt for i, stmt in enumerate(tree.body[0].body) if i in statements]
    module = compile(tree, method.__code__.co_filename, "exec", dont_inherit=True)
    selected = FunctionType(module.co_consts[0], method.__globals__)
    case.test_max_nodes = MethodType(selected, case)
