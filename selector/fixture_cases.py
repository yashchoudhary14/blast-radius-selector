"""Compare the values that the pinned benchmark methods use from SQL fixtures."""

from runner.query_cases import EXECUTOR, OPTIMIZER

from .source_equivalence import compile_source, line_insertion, relocated_code, same_code, same_directives


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


def unaffected_fixture_tests(analyzer, edits):
    fixtures = edits.keys() & (OPTIMIZER_READERS.keys() | {TPC_H})
    if len(fixtures) != 1:
        return 0
    path = next(iter(fixtures))
    source = edits[path]
    if source is None:
        return 0
    file = analyzer.base / path
    if not file.is_file():
        return 0
    mode = file.stat().st_mode & 0o777
    if edits.modes.get(path, mode) != mode:
        return 0
    try:
        before, after = pairs(file.read_bytes()), pairs(source)
    except (ValueError, IndexError):
        return 0

    safe = set()
    if path in {TPC_H, TPC_DS}:
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
    if path in OPTIMIZER_READERS and before == after:
        safe.add(OPTIMIZER + OPTIMIZER_READERS[path])
    unaffected = sum(1 << analyzer.num[node] for node in safe if node in analyzer.num)
    for other in edits.keys() - {path}:
        # A companion must leave the fixture reader's code and inputs intact.
        unaffected &= ~analyzer.index["reads"].get(other, 0)
        if not unaffected or not companion_equivalent(analyzer, edits, other):
            return 0
        if other.endswith(".py"):
            unaffected &= ~analyzer.location_observers(all_dynamic=True)
    return unaffected


def companion_equivalent(analyzer, edits, path):
    file = analyzer.base / path
    source = edits[path]
    if source is None or not file.is_file() or edits.modes.get(path, file.stat().st_mode & 0o777) != file.stat().st_mode & 0o777:
        return False
    if path.startswith(FIXTURES) and path.endswith(".sql"):
        # Each benchmark reads an explicit fixture. Independent data readers
        # of this companion stay selected through the read mask above.
        return True
    if not path.startswith("sqlglot/") or not path.endswith(".py"):
        return False
    before = file.read_bytes()
    original, diagnostics = analyzer.old_compiled(path)
    revised, new_diagnostics = compile_source(source, path)
    if diagnostics or new_diagnostics:
        return False
    insertion = line_insertion(before, source)
    return ((same_code(original, revised) or (insertion and relocated_code(original, revised, insertion)))
            and same_directives(before, source, insertion))
