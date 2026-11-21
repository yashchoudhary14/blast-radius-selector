"""Attribute the intact fixture loop, including warm pool workers, to query indices."""

import atexit
import json
import os
import sys
import threading
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

index = -1


def activate(i):
    global index
    index = i
    return invoke


def invoke(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    finally:
        activate(-1)  # result serialization and pool shutdown are shared


class Task:
    def __init__(self, i):
        self.i = i

    def __reduce__(self):
        # Tag before the worker unpickles the SQLGlot expression arguments.
        return activate, (self.i,)


def install(directory):
    # Use a shared bucket for the parent pool's background pickling threads.
    rows = defaultdict(lambda: defaultdict(set))
    main = threading.get_ident()

    def line(code, number):
        path = code.co_filename
        if not path.startswith(("/sqlglot/sqlglot/", "/sqlglot/tests/")):
            return sys.monitoring.DISABLE
        context = index if threading.get_ident() == main else -1
        rows[context][path[9:]].add(number)

    tool = 4
    dest = Path(directory) / str(os.getpid())
    dest.touch()
    sys.monitoring.use_tool_id(tool, "full fixture loop")
    sys.monitoring.register_callback(tool, sys.monitoring.events.LINE, line)
    sys.monitoring.set_events(tool, sys.monitoring.events.LINE)

    def save():
        sys.monitoring.set_events(tool, 0)
        dest.write_text(json.dumps({i: {p: sorted(v) for p, v in paths.items()}
                                    for i, paths in rows.items()}))

    atexit.register(save)


def instrument(case, module, node):
    from runner.query_cases import EXECUTOR, QUERIES

    method = getattr(case, case._testMethodName)
    futures = {}
    submit = ProcessPoolExecutor.submit

    def tagged_submit(pool, function, /, *args, **kwargs):
        assert index >= 0
        future = submit(pool, Task(index), function, *args, **kwargs)
        futures[future] = index
        return future

    def fixtures(rows):
        for i, row in enumerate(rows):
            activate(i)
            yield row
        activate(-1)

    def wrapped():
        from contextlib import ExitStack
        with ExitStack() as stack:
            stack.enter_context(patch.object(ProcessPoolExecutor, "submit", tagged_submit))
            if node.startswith(EXECUTOR):
                class Rows(list):
                    def __iter__(self):
                        return fixtures(super().__iter__())

                attr = QUERIES[node] + "_sqls"
                stack.enter_context(patch.object(case, attr, Rows(getattr(case, attr))))
                helper = case.subtestHelper

                def tagged_helper(i, *args, **kwargs):
                    activate(i)
                    try:
                        return helper(i, *args, **kwargs)
                    finally:
                        activate(-1)

                stack.enter_context(patch.object(case, "subtestHelper", tagged_helper))
            else:
                load, completed = module.load_sql_fixture_pairs, module.as_completed
                stack.enter_context(patch.object(module, "load_sql_fixture_pairs",
                                                 lambda name: fixtures(load(name))))

                def tagged_completed(items):
                    for future in completed(items):
                        activate(futures[future])
                        yield future
                    activate(-1)

                stack.enter_context(patch.object(module, "as_completed", tagged_completed))
            return method()

    setattr(case, case._testMethodName, wrapped)
