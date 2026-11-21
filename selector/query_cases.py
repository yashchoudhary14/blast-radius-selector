"""Restrict expensive fixture loops using their pristine per-case execution."""

import ast
import gzip
import json
import marshal
from bisect import bisect_left
from functools import lru_cache
from types import CodeType

from runner.query_cases import EXECUTOR


MAX_NODES = "tests/test_parser.py::TestParser::test_max_nodes"


def dead_condition(analyzer, edits, files):
    """An unreached if-name negation changes neither bindings nor class data."""
    if len(edits) != 1:
        return False
    path, source = next(iter(edits.items()))
    if not (source and path.startswith("sqlglot/") and path.endswith(".py")
            and (analyzer.base / path).is_file()):
        return False
    old, _ = analyzer.old_tree_and_lines(path)
    new = ast.parse(source)
    before = [node for node in ast.walk(old) if isinstance(node, ast.If)]
    after = [node for node in ast.walk(new) if isinstance(node, ast.If)]
    if len(before) != len(after):
        return False
    changed = False
    for left, right in zip(before, after):
        a, b = left.test, right.test
        if type(a) is type(b):
            continue
        plain, negated = (a, b) if isinstance(a, ast.Name) else (b, a)
        if not (isinstance(plain, ast.Name) and isinstance(negated, ast.UnaryOp)
                and isinstance(negated.op, ast.Not) and isinstance(negated.operand, ast.Name)
                and plain.id == negated.operand.id and left.lineno == a.lineno == a.end_lineno):
            return False
        if (left.lineno in analyzer.line_hits(path)
                or left.lineno in analyzer.indirect_hits.get(path, {})
                or any(a <= left.lineno <= b for a, b, _ in files.get(path, ()))):
            return False
        right.test = a
        changed = True
    return changed and ast.dump(old) == ast.dump(new)


def frame_layout(code):
    # An edit in an untaken branch can still make a name local, create a
    # closure or turn the entire function into a generator.
    return (code.co_name, code.co_flags, code.co_varnames, code.co_freevars, code.co_cellvars,
            tuple(frame_layout(c) for c in code.co_consts if isinstance(c, CodeType)))


@lru_cache(maxsize=32)
def pinned_layout(base, path):
    return frame_layout(compile((base / path).read_bytes(), path, "exec", dont_inherit=True))


def body_spans(old, new, spans):
    """Use statement precision for expression edits, widen changed control flow."""
    if not old or not new or ast.get_docstring(old, clean=False) != ast.get_docstring(new, clean=False):
        return None

    def structure(fn):
        return [(type(s), s.col_offset, len(getattr(s, "body", ())), len(getattr(s, "orelse", ())))
                for s in ast.walk(fn) if isinstance(s, ast.stmt)]

    if len(old.body) == 1 and ast.get_docstring(old) is not None:
        return None
    if spans and structure(old) == structure(new):
        return spans
    # For insertions, deletion and restructured blocks, any old invocation
    # may reach the edit. Exclude the def line (executed at import).
    return [(old.body[0].lineno, old.end_lineno)]


class QueryCases:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        bin_path = analyzer.maps / "query_cases.bin"
        if bin_path.is_file():
            data = marshal.loads(bin_path.read_bytes())
        else:
            with gzip.open(analyzer.maps / "query_cases.json.gz") as source:
                data = json.load(source)
        self.maps = {analyzer.num[node]: row for node, row in data.items() if node in analyzer.num}
        self._tpcds_weights = None

    def _get_tpcds_weights(self):
        if self._tpcds_weights is None:
            fixture = self.analyzer.base / "tests/fixtures/optimizer/tpc-ds/tpc-ds.sql"
            if fixture.is_file():
                lines = [ln for ln in fixture.read_text(encoding="utf-8").splitlines()
                         if ln and not ln.startswith("--")]
                stmts = "\n".join(lines).split(";")
                weights = [1.0 if "# execute: true" in stmts[i] else 0.04
                           for i in range(0, len(stmts) - 1, 2)]
                self._tpcds_weights = (weights, sum(weights))
            else:
                self._tpcds_weights = ([], 0.0)
        return self._tpcds_weights

    def subsets(self, candidates, edits):
        analyzer = self.analyzer
        spans = candidates.query_spans
        if candidates.broken:
            return {}
        if not spans:
            n = analyzer.num.get(MAX_NODES)
            if (n in self.maps and candidates.mask & (1 << n)
                    and not any(analyzer.index["reads"].get(p, 0) & (1 << n) for p in edits)
                    and dead_condition(analyzer, edits, self.maps[n][2])):
                return {n: []}
            return {}
        for path, intervals in spans.items():
            imports = analyzer.indirect_hits.get(path, {})
            if any(line in imports for start, end in intervals for line in range(start, end + 1)):
                return {}
        subsets = {}
        for number in self.maps:
            if not candidates.mask & (1 << number):
                continue
            if any(mask & (1 << number) for mask in candidates.isolated_test_paths.values()):
                continue
            if any(analyzer.index["reads"].get(path, 0) & (1 << number) for path in edits):
                continue
            count, masks, files = self.maps[number]
            hit = 0
            for path, intervals in spans.items():
                blocks = files.get(path, ())
                for start, end in intervals:
                    i = bisect_left(blocks, start, key=lambda row: row[1])
                    while i < len(blocks) and blocks[i][0] <= end:
                        hit |= masks[blocks[i][2]]
                        i += 1
            if hit.bit_count() < count:
                subsets[number] = [i for i in range(count) if hit & (1 << i)]
        return subsets

    def body_cost(self, n, indices, execution):
        # Optimizer pool creation is already included in Execution.setup().
        node = execution.ids[n]
        pool = .65 if node.startswith(EXECUTOR + "test_execute_") else 0.
        if node == EXECUTOR + "test_execute_tpcds":
            weights, total = self._get_tpcds_weights()
            if total > 0 and len(weights) == self.maps[n][0]:
                fraction = sum(weights[i] for i in indices) / total
                return pool + max(0., execution.body[n] - pool) * fraction
        fraction = len(indices) / self.maps[n][0]
        return pool + max(0., execution.body[n] - pool) * fraction
