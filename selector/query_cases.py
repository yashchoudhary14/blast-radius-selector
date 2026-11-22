"""Restrict expensive fixture loops using their pristine per-case execution."""

import ast
import gzip
import json
import marshal
from bisect import bisect_left
from types import CodeType

from runner.parser_cases import MAX_NODES
from runner.query_cases import EXECUTOR

from .generator_modifiers import RENDER_MODIFIERS, render_modifier_edit, render_modifier_read_lines
from .query_bindings import bindings, reference_lines, removed_private_functions


TPCDS = EXECUTOR + "test_execute_tpcds"


def reached_cases(masks, files, path, intervals):
    hit = 0
    blocks = files.get(path, ())
    for start, end in intervals:
        i = bisect_left(blocks, start, key=lambda row: row[1])
        while i < len(blocks) and blocks[i][0] <= end:
            hit |= masks[blocks[i][2]]
            i += 1
    return hit


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


def frame_layout(code, *, locals=True):
    # An edit in an untaken branch can still make a name local, create a
    # closure or turn the entire function into a generator.
    return (code.co_name, code.co_flags, code.co_varnames if locals else (), code.co_freevars, code.co_cellvars,
            tuple(frame_layout(c, locals=locals) for c in code.co_consts if isinstance(c, CodeType)))


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

    def binding_readers(self, path, edits, eligible):
        analyzer = self.analyzer
        original = analyzer.base / path
        source = edits[path]
        if not (source and original.is_file() and path.startswith(("sqlglot/", "tests/"))
                and path.endswith(".py") and path != "sqlglot/__init__.py"):
            return None
        if path in edits.modes and edits.modes[path] != original.stat().st_mode & 0o777:
            return None
        before = original.read_bytes()
        result = bindings(before, source, path)
        render_modifiers = result is None and render_modifier_edit(before, source, path)
        removed = removed_private_functions(before, source) if result is None and not render_modifiers else set()
        if result is None and not render_modifiers and not removed:
            return None
        values, fresh = result if result is not None else ({RENDER_MODIFIERS} if render_modifiers else set(), set())
        if any(not name.startswith("_") for name in fresh):
            # A new name can also appear through a package's star re-export.
            mod = path[:-3].replace("/", ".").removesuffix(".__init__")
            if path.startswith("sqlglot/expressions/") or mod in analyzer._test_star_imports():
                return None
        observers = analyzer.location_observers(all_dynamic=True) | analyzer.state_refs.mask("__annotations__")
        if path.startswith(("sqlglot/dialects/", "sqlglot/optimizer/", "sqlglot/expressions/")):
            # The corresponding lazy packages and serde can resolve names by string.
            observers |= analyzer.state_refs.mask("*module")
        readers = eligible & observers
        for name in values | fresh | removed:
            for ref_path in analyzer.state_refs.refs.get(name, ()):
                lines = (render_modifier_read_lines(analyzer, ref_path) if render_modifiers
                         else reference_lines(analyzer, ref_path, name, name in values))
                if not lines:
                    continue
                for number in self.maps:
                    bit = 1 << number
                    if eligible & bit and not readers & bit:
                        _, masks, files = self.maps[number]
                        if reached_cases(masks, files, ref_path, ((line, line) for line in lines)):
                            readers |= bit
        return readers

    def _get_tpcds_weights(self):
        if self._tpcds_weights is None:
            calibration = self.analyzer.maps / "tpcds-case-cpu.json"
            if calibration.is_file():
                # The host process clock rounds short queries down to zero.
                weights = [max(8, ms) for ms in json.loads(calibration.read_bytes())]
                self._tpcds_weights = (weights, sum(weights))
                return self._tpcds_weights
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
        if candidates.broken:
            return {}
        spans = {path: rows for path, rows in candidates.query_spans.items() if rows}
        opaque = {path for path, rows in candidates.query_spans.items() if rows is None}
        unresolved = opaque - candidates.query_guarded
        blocked = candidates.query_blocked
        proved_bindings = set()
        heavy_mask = sum(1 << n for n in self.maps if analyzer.ids[n] in (TPCDS, MAX_NODES))
        traced = sum(1 << n for n in self.maps) & candidates.mask & ~blocked
        if traced and not candidates.structural:
            for path in sorted(unresolved):
                readers = self.binding_readers(path, edits, traced)
                if readers is not None:
                    blocked |= readers
                    proved_bindings.add(path)
            unresolved -= proved_bindings
        if not spans and not proved_bindings:
            n = analyzer.num.get(MAX_NODES)
            if (n in self.maps and candidates.mask & (1 << n)
                    and not candidates.query_blocked & (1 << n)
                    and not any(analyzer.index["reads"].get(p, 0) & (1 << n) for p in edits)
                    and dead_condition(analyzer, edits, self.maps[n][2])):
                return {n: []}
            return {}
        for path, intervals in spans.items():
            if path in edits.modes and edits.modes[path] != (analyzer.base / path).stat().st_mode & 0o777:
                return {}
            imports = analyzer.indirect_hits.get(path, {})
            for start, end in intervals:
                for line in range(start, end + 1):
                    blocked |= imports.get(line, 0)
        subsets = {}
        for number in self.maps:
            bit = 1 << number
            if not (candidates.mask & ~blocked) & bit:
                continue
            if any(analyzer.index["reads"].get(path, 0) & bit for path in edits):
                continue
            count, masks, files = self.maps[number]
            is_heavy = bool(bit & heavy_mask)
            if not is_heavy and (unresolved or candidates.query_heavy_only):
                continue
            if opaque and (candidates.structural or any(
                path in files or (analyzer.importers[path] | analyzer.index["all_hits"].get(path, 0)) & bit
                for path in unresolved
            )):
                continue
            hit = 0
            for path, intervals in spans.items():
                hit |= reached_cases(masks, files, path, intervals)
            if hit.bit_count() < count:
                subsets[number] = [i for i in range(count) if hit & (1 << i)]
        return subsets

    def body_cost(self, n, indices, execution):
        # Optimizer pool creation is already included in Execution.setup().
        node = execution.ids[n]
        if node == MAX_NODES and self.maps[n][0] == 4:
            # The two wide inputs dominate CPU. The smoke and reset groups
            # use less than 0.02 host CPU seconds each.
            weights = (.02, 23.83, 8.31, .02)
            return execution.body[n] * sum(weights[i] for i in indices) / sum(weights)
        pool = .65 if node.startswith(EXECUTOR + "test_execute_") else 0.
        if node == TPCDS:
            weights, total = self._get_tpcds_weights()
            if total > 0 and len(weights) == self.maps[n][0]:
                fraction = sum(weights[i] for i in indices) / total
                return pool + max(0., execution.body[n] - pool) * fraction
        fraction = len(indices) / self.maps[n][0]
        return pool + max(0., execution.body[n] - pool) * fraction
