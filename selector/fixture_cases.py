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
PROJECTED_FIXTURES = OPTIMIZER_READERS.keys() | {TPC_H}


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


def fixture_effects(analyzer, edits, other_mask, revisions):
    """Return safe whole-test omissions and changed query indices."""
    fixtures = edits.keys() & PROJECTED_FIXTURES
    if len(fixtures) != 1:
        return 0, {}
    path = next(iter(fixtures))
    file = analyzer.base / path
    source = edits[path]
    if source is None or not file.is_file() or not same_mode(file, edits, path):
        return 0, {}

    keep = other_mask
    for other in edits.keys() - fixtures:
        if not companion_equivalent(analyzer, edits, other, revisions.get(other)):
            return 0, {}
        keep |= analyzer.index["reads"].get(other, 0)
        if other.endswith(".py"):
            keep |= analyzer.location_observers(all_dynamic=True)
    try:
        before, after = pairs(file.read_bytes()), pairs(source)
    except (ValueError, IndexError):
        return 0, {}

    whole = set()
    subsets = {}

    def project(node, left, right, *, omit_equal=True):
        num = analyzer.num.get(node)
        if num is None or keep & (1 << num):
            return
        if omit_equal and left == right:
            whole.add(node)
        elif len(left) == len(right):
            changed = [i for i, (old, new) in enumerate(zip(left, right)) if old != new]
            if len(changed) < len(left):
                subsets[num] = changed

    if path in {TPC_H, TPC_DS}:
        # Shared setup loads both fixtures. Each benchmark consumes only its
        # own family after setup succeeds.
        whole.add(EXECUTOR + "test_agg_order")
        if path == TPC_DS:
            whole.update((EXECUTOR + "test_execute_tpch", EXECUTOR + "test_optimized_tpch"))
            project(EXECUTOR + "test_execute_tpcds", executor_inputs(before), executor_inputs(after))
            project(OPTIMIZER + "test_tpcds", before, after)
        else:
            whole.add(EXECUTOR + "test_execute_tpcds")
            project(EXECUTOR + "test_execute_tpch", executor_inputs(before), executor_inputs(after))
            project(EXECUTOR + "test_optimized_tpch", [r[1:] for r in before], [r[1:] for r in after])
            project(OPTIMIZER + "test_tpch", before, after, omit_equal=False)
    elif before == after:
        whole.add(OPTIMIZER + OPTIMIZER_READERS[path])
    return sum(1 << analyzer.num[node] for node in whole if node in analyzer.num) & ~keep, subsets


def same_mode(file, edits, path):
    mode = file.stat().st_mode & 0o777
    return edits.modes.get(path, mode) == mode


def companion_equivalent(analyzer, edits, path, revision):
    file = analyzer.base / path
    source = edits[path]
    if source is None or not file.is_file() or not same_mode(file, edits, path):
        return False
    if path.startswith(FIXTURES) and path.endswith(".sql"):
        # The pinned methods open specific fixtures. Keep independent readers
        # of this file through their observed masks.
        return True
    if not path.startswith("sqlglot/") or not path.endswith(".py"):
        return False
    before = file.read_bytes()
    original, diagnostics = analyzer.old_compiled(path)
    if diagnostics:
        return False
    if revision is None:
        revised, diagnostics = compile_source(source, path)
        if diagnostics:
            return False
    else:
        # Analyzer caches a revision only after rejecting changed diagnostics.
        revised = revision[1]
    insertion = line_insertion(before, source)
    return ((same_code(original, revised) or (insertion and relocated_code(original, revised, insertion)))
            and same_directives(before, source, insertion))
