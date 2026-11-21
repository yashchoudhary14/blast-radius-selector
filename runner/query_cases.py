"""Restrict the pinned TPC methods at their fixture input, after shared setup."""

from functools import wraps
from unittest.mock import patch


EXECUTOR = "tests/test_executor.py::TestExecutor::"
OPTIMIZER = "tests/test_optimizer.py::TestOptimizer::"
QUERIES = {
    EXECUTOR + "test_execute_tpcds": "tpcds",
    EXECUTOR + "test_execute_tpch": "tpch",
    EXECUTOR + "test_optimized_tpch": "tpch",
    OPTIMIZER + "test_tpcds": "tpcds",
    OPTIMIZER + "test_tpch": "tpch",
}


def filter_queries(case, module, node, indices):
    wanted = frozenset(indices)
    if any(type(i) is not int or i < 0 for i in wanted):
        raise ValueError("expected zero-based fixture indices")

    def select(rows):
        rows = list(rows)
        if wanted and max(wanted) >= len(rows):
            raise ValueError("query case outside pinned fixture")
        return [row for i, row in enumerate(rows) if i in wanted]

    method = getattr(case, case._testMethodName)

    @wraps(method)
    def filtered():
        if node.startswith(EXECUTOR):
            # subtestHelper indexes the same list as _mp_execute. Shadow it
            # on this instance so future i and its expected SQL stay paired.
            attr = QUERIES[node] + "_sqls"
            context = patch.object(case, attr, select(getattr(case, attr)))
        else:
            load = module.load_sql_fixture_pairs
            context = patch.object(module, "load_sql_fixture_pairs", lambda name: select(load(name)))
        with context:
            return method()

    setattr(case, case._testMethodName, filtered)
