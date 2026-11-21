"""Compare the values that the pinned benchmark methods use from SQL fixtures."""

from runner.query_cases import EXECUTOR, OPTIMIZER


FIXTURES = "tests/fixtures/optimizer/"
TPC_H = FIXTURES + "tpc-h/tpc-h.sql"
TPC_DS = FIXTURES + "tpc-ds/tpc-ds.sql"
OPTIMIZER_READERS = {
    TPC_DS: "test_tpcds",
    FIXTURES + "merge_subqueries.sql": "test_merge_subqueries",
    FIXTURES + "qualify_columns.sql": "test_qualify_columns",
    FIXTURES + "qualify_columns_ddl.sql": "test_qualify_columns",
    FIXTURES + "simplify.sql": "test_simplify",
}


def pairs(source):
    # Match tests.helpers.load_sql_fixture_pairs, including its error on a
    # metadata-only SQL statement. The caller retains tests on a loader error.
    lines = source.decode("utf-8").splitlines()
    statements = iter("\n".join(line for line in lines if line and not line.startswith("--")).split(";"))
    rows = []
    for sql, expected in zip(statements, statements):
        lines = sql.strip().split("\n")
        i, meta = 0, {}
        while lines[i].startswith("#"):
            key, value = lines[i].split(":", 1)
            meta[key.lstrip("#").strip()] = value.strip()
            i += 1
        rows.append((tuple(meta.items()), "\n".join(lines[i:]), expected.strip()))
    return rows


def executor_inputs(rows):
    # The parent parses every SQL string. The worker only uses the truthiness
    # of the execute option, and the assertion never reads the expected SQL.
    return [(sql, bool(dict(meta).get("execute"))) for meta, sql, _ in rows]


def _changed_fixture_pairs(analyzer, edits):
    found = {}
    for path, source in edits.items():
        if source is None or (path not in OPTIMIZER_READERS and path != TPC_H):
            continue
        file = analyzer.base / path
        if not file.is_file():
            return None
        mode = file.stat().st_mode & 0o777
        if edits.modes.get(path, mode) != mode:
            return None
        try:
            before, after = pairs(file.read_bytes()), pairs(source)
        except (ValueError, IndexError):
            return None
        found[path] = (before, after)
    return found


def unaffected_fixture_tests(analyzer, edits, *, other_mask=0):
    if other_mask:
        return 0
    non_fixtures = [p for p in edits if p not in OPTIMIZER_READERS and p != TPC_H]
    if non_fixtures and any(p not in (TPC_DS, TPC_H) for p in edits):
        # Allow auxiliary edits (e.g. comment-only code edits) only alongside TPC_DS/TPC_H
        # when those auxiliary edits contributed zero selected tests.
        pass
    elif len(edits) != 1:
        return 0
    parsed = _changed_fixture_pairs(analyzer, edits)
    if not parsed:
        return 0
    if non_fixtures and any(p not in {TPC_DS, TPC_H} for p in parsed):
        return 0

    safe = set()
    tpc_paths = set(parsed) & {TPC_H, TPC_DS}
    if tpc_paths == {TPC_H} or tpc_paths == {TPC_DS}:
        path = next(iter(tpc_paths))
        before, after = parsed[path]
        # Shared setup loads both fixtures into separate class attributes.
        # Each benchmark consumes only its own family after setup succeeds.
        safe.add(EXECUTOR + "test_agg_order")
        same_inputs = executor_inputs(before) == executor_inputs(after)
        if path == TPC_H or same_inputs:
            safe.add(EXECUTOR + "test_execute_tpcds")
        if path == TPC_DS or same_inputs:
            safe.add(EXECUTOR + "test_execute_tpch")
        if path == TPC_DS or [r[1:] for r in before] == [r[1:] for r in after]:
            safe.add(EXECUTOR + "test_optimized_tpch")
        if path == TPC_DS and before == after:
            safe.add(OPTIMIZER + "test_tpcds")
    elif len(edits) == 1 and len(parsed) == 1:
        path, (before, after) = next(iter(parsed.items()))
        if path in OPTIMIZER_READERS and before == after:
            safe.add(OPTIMIZER + OPTIMIZER_READERS[path])
    return sum(1 << analyzer.num[node] for node in safe if node in analyzer.num)


def fixture_case_subsets(analyzer, edits, *, other_mask=0):
    parsed = _changed_fixture_pairs(analyzer, edits)
    if not parsed:
        return {}
    tpc_paths = set(parsed) & {TPC_H, TPC_DS}
    if len(tpc_paths) != 1:
        return {}
    path = next(iter(tpc_paths))
    if any(p not in {TPC_DS, TPC_H} for p in parsed):
        return {}
    before, after = parsed[path]
    if len(before) != len(after):
        return {}
    subsets = {}

    def add(node, changed):
        num = analyzer.num.get(node)
        if num is None or (other_mask & (1 << num)):
            return
        if len(changed) < len(before):
            subsets[num] = changed

    if path == TPC_DS:
        add(
            EXECUTOR + "test_execute_tpcds",
            [i for i, (b, a) in enumerate(zip(executor_inputs(before), executor_inputs(after))) if b != a],
        )
        add(
            OPTIMIZER + "test_tpcds",
            [i for i, (b, a) in enumerate(zip(before, after)) if b != a],
        )
    else:
        add(
            EXECUTOR + "test_execute_tpch",
            [i for i, (b, a) in enumerate(zip(executor_inputs(before), executor_inputs(after))) if b != a],
        )
        add(
            EXECUTOR + "test_optimized_tpch",
            [i for i, (b, a) in enumerate(zip(before, after)) if b[1:] != a[1:]],
        )
        add(
            OPTIMIZER + "test_tpch",
            [i for i, (b, a) in enumerate(zip(before, after)) if b != a],
        )
    return subsets
