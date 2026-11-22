"""Compare measured runner CPU with the marginal cost of the emitted worker schedule."""

import os
import re
import time
from collections import defaultdict
from functools import lru_cache

from runner import GENERATOR_MEMORY, cpu_time
from .analyze import numbers
from .dialect_cases import DialectCases
from .patch_proof import PatchProof
from .proxies import can_prove_pass
from .query_cases import MAX_NODES, TPCDS
from .workers import WORKER_MS, schedule


# Shared fork and unittest fixture costs for a batch of warm test methods.
STARTUP = 0.015
PER_SECOND = 1.001
PER_MODULE = 0.002
FIXTURE = {"tests/test_executor.py": 1.173, "tests/test_optimizer.py": 0.095}
# Pristine single-test child CPU under the same limits, inclusive of collection.
SINGLE = {
    773: .675, 1082: 1.711, 1106: 39.413, 1107: 2.308, 1123: 5.635,
    1224: 3.034, 1295: 1.843, 1313: .878, 1343: 2.065, 1358: .959,
    1361: 1.653, 1382: 3.177, 1415: 1.115, 1428: 3.091, 1434: 2.363,
    1440: 8.075, 1490: 33.594, 1541: .459, 1560: 1.004, 1561: .600,
}
PROOF_CPU = {1082: .3, 1106: 4.5, 1107: .4, 1123: .5, 1361: 1.4}
WITNESS_EDIT = re.compile(r"^\+(?!\+\+).*?(?:\bif\b|\bTrue\b|\bFalse\b|==|!=)", re.MULTILINE)


class Execution:
    def __init__(self, analyzer, tests, millis):
        self.analyzer, self.millis = analyzer, millis
        self.ids = {t["number"]: t["id"] for t in tests}
        self.num = {node: n for n, node in self.ids.items()}
        self.memory = self.num.get(GENERATOR_MEMORY)
        self.catalog_workers, catalog_ms = schedule(millis, millis)
        self.catalog_cost = catalog_ms / 1000
        self.heavy_cases = {self.num[test] for test in (MAX_NODES, TPCDS) if test in self.num}
        self.dialects = DialectCases(analyzer.maps)
        self.files = {t["number"]: t["file"] for t in tests}
        self.pool_tests = set(numbers(analyzer.index["by_func"].get("tests/test_optimizer.py", {}).get(
            "TestOptimizer.check_file", 0)))
        self.import_cpu = 0.
        self.body = {n: PER_SECOND * ms / 1000 for n, ms in millis.items()}
        for n, total in SINGLE.items():
            if n in self.body:
                self.body[n] = max(0, total - self.setup({n}))

    def setup(self, batch):
        files = {self.files[n] for n in batch}
        # The optimizer spawns a fresh Python worker on its first submitted
        # fixture pair. Other optimizer methods only pay for the DuckDB setup.
        pool = .65 if self.pool_tests.intersection(batch) else 0.
        return STARTUP + PER_MODULE * len(files) + sum(FIXTURE.get(f, 0) for f in files) + pool

    def run_cost(self, batch):
        return self.import_cpu + self.setup(batch) + sum(self.body[n] for n in batch)

    def cost(self, selected):
        return schedule(selected, self.millis)[1] / 1000

    def apply_witnesses(self, proof, eligible, selected, commit, log):
        for n in sorted(eligible, key=self.millis.get, reverse=True):
            start = cpu_time()
            reason = proof.witness(n)
            if reason:
                selected.remove(n)
            log(f"proof {commit} n={n} passed={int(bool(reason))} reason={reason or 'no-witness'} "
                f"cpu={cpu_time()-start:.6f}")

    def batches(self, pending, filtered):
        cheap = {n for n in pending if self.millis[n] < 500}
        heavy = pending - cheap
        groups = defaultdict(set)
        for n in pending:
            groups[self.files[n]].add(n)
        yield pending
        yield cheap
        # Some tests cost more to run than their entire worker load. Price the
        # rest together so shared setup and the last worker fee count only once.
        yield cheap | {n for n in heavy if self.body[n] < self.millis[n] / 1000}
        paired = pending & self.heavy_cases & filtered.keys()
        if len(paired) > 1:
            yield paired
            yield paired | cheap
        for n in sorted(heavy):
            yield {n}
        yield from groups.values()

    def select(self, get_session, commit, patch, candidates, deadline, log):
        selected = candidates.selected()
        subsets = candidates.query_subsets
        for n, indices in subsets.items():
            if not indices:
                log(f"queries {commit} n={n} cases=0 outcome=pass")
        if not selected or candidates.broken:
            return selected
        failed, observed = set(), set()
        protected = set()
        potential_observers = set()
        for path in candidates.python_paths:
            potential_observers.update(numbers(self.analyzer.index["heap_observers"] & self.analyzer.importers[path]))
        def conservative(remaining):
            # A pending failure can affect heap measurements in the full suite.
            # A passing probe only settles the methods it actually completed.
            # Variant: no suite-wide heap observers (SELECTOR_SUITE_OBSERVERS=1 restores them).
            if os.getenv("SELECTOR_SUITE_OBSERVERS") == "1" and remaining - potential_observers:
                return remaining | potential_observers
            return remaining

        # A witness that skips companion files also needs the whole-test guard.
        eligible = {n for n in PROOF_CPU.keys() & selected if not candidates.query_blocked & (1 << n)}
        if get_session is None or time.monotonic() >= deadline:
            if eligible and not candidates.structural and WITNESS_EDIT.search(patch):
                proof = PatchProof(self.analyzer, patch, candidates.isolated_test_paths)
                self.apply_witnesses(proof, eligible, selected, commit, log)
            return conservative(selected)
        if len(selected) == 1 and not candidates.python_paths:
            # A single direct data reader can save only its own worker on a pass.
            # Keep the reader when its expected edit is the sole evidence.
            return selected
        stage = get_session().import_stage(candidates.python_paths)
        self.import_cpu = {2: 0., 1: .025, 0: .065, -1: .15}[stage]
        discounts = {n: self.body[n] - self.analyzer.queries.body_cost(n, indices, self)
                     for n, indices in subsets.items() if indices}

        def run_cost(batch):
            return self.run_cost(batch) - sum(discounts[n] for n in discounts if n in batch)

        uncandidated = frozenset(n for n in self.millis if not candidates.mask & (1 << n))

        @lru_cache(maxsize=64)
        def emitted_cost_cached(selection):
            cost = self.cost(selection)
            if candidates.structural or cost < self.catalog_cost:
                return cost
            # The final lane reuses the catalog's worker assignment at the
            # price cap, but it still honors completed and zero-case probes.
            emitted = selection | uncandidated
            loads = [sum(self.millis[n] for n in worker if n in emitted) for worker in self.catalog_workers
                     if any(n in emitted for n in worker)]
            return (max(loads) + WORKER_MS * len(loads)) / 1000

        def emitted_cost(selection):
            return emitted_cost_cached(frozenset(selection))

        subset = self.dialects.subset(candidates) if self.memory is not None else None
        if subset is not None:
            subset_cost = self.import_cpu + self.setup({self.memory}) + self.dialects.body_cost(subset)
        memory_passed = False

        def include_importers():
            for path in candidates.python_paths:
                selected.update(numbers(self.analyzer.importers[path] | self.analyzer.index["all_hits"].get(path, 0)))
                candidates.reasons.append(("probe-import-error", path))
            if memory_passed:
                selected.discard(self.memory)

        def probe_memory():
            # A standalone test includes collection and stops at the first
            # failing dialect, even when other probes already found failures.
            nonlocal memory_passed
            budget = deadline - time.monotonic()
            if subset_cost > budget:
                return
            start = cpu_time()
            result = get_session().run(patch, [GENERATOR_MEMORY], stop_on_first_fail=True,
                                       overall_timeout=budget, case_filter={GENERATOR_MEMORY: subset})
            passed = (result.ran == [GENERATOR_MEMORY] and not result.failing
                      and not result.import_error and "incomplete child run" not in result.notes)
            log(f"dialects {commit} cases={len(subset)} passed={int(passed)} "
                f"est={subset_cost:.3f} cpu={cpu_time()-start:.6f}")
            if passed:
                memory_passed = True
                selected.discard(self.memory)
                potential_observers.discard(self.memory)
                protected.discard(self.memory)
            else:
                selected.add(self.memory)
                protected.add(self.memory)
                if result.failing:
                    failed.add(self.memory)
                if result.import_error:
                    include_importers()
            observed.add(self.memory)

        if subset is not None and self.memory in selected:
            kept = conservative(selected)
            if .5 * (emitted_cost(kept) - emitted_cost(kept - {self.memory})) > subset_cost:
                probe_memory()
        proof = root = None
        tried_replay = set()
        filtered_heavy = self.heavy_cases & discounts.keys()
        promoted = set()

        def rank(n):
            level = -1 if n in promoted else 0 if candidates.priority & (1 << n) else 1 if candidates.direct & (1 << n) else 2
            return level, run_cost([n]), n

        def run(batch, early):
            order = sorted(batch, key=rank)
            start = cpu_time()
            budget = deadline - time.monotonic()
            if budget <= 0:
                return False
            filters = {self.ids[n]: subsets[n] for n in batch if n in discounts}
            result = get_session().run(patch, [self.ids[n] for n in order], case_filter=filters,
                                       stop_on_first_fail=early or bool(filters), overall_timeout=budget)
            bad = {self.num[node] for node in result.failing}
            failed.update(bad)
            if bad and os.getenv("SELECTOR_SUITE_OBSERVERS") == "1":
                protected.update(potential_observers)
            if result.import_error:
                include_importers()
            complete = "incomplete child run" not in result.notes
            ran = {self.num[node] for node in result.ran}
            for n in ran & discounts.keys():
                outcome = "unknown" if result.notes else "fail" if n in bad else "pass"
                if outcome == "unknown":
                    protected.add(n)
                log(f"queries {commit} n={n} cases={len(subsets[n])} outcome={outcome}")
            if complete and not result.import_error:
                observed.update(ran)
                selected.difference_update(ran - bad)
            selected.update(protected)
            log(f"run {commit} attempted={len(ran)}/{len(order)} failures={len(bad)} cpu={cpu_time()-start:.6f} "
                f"child={result.cpu_s:.6f} recorded={sum(self.millis[n] for n in ran)/1000:.3f} "
                f"modules={len({self.files[n] for n in ran})} complete={int(complete)}")
            return complete and not result.import_error and bool(ran)

        if eligible:
            proof = PatchProof(self.analyzer, patch, candidates.isolated_test_paths)
            self.apply_witnesses(proof, eligible, selected, commit, log)

        def roi_cost(s):
            c = emitted_cost(s)
            if (subset is not None and self.memory in s and self.memory not in selected
                    and self.memory not in observed and self.memory not in failed):
                c_no_mem = emitted_cost(s - {self.memory})
                if .5 * (c - c_no_mem) > subset_cost:
                    return c_no_mem + subset_cost + 0.1 * (c - c_no_mem)
            return c

        while time.monotonic() < deadline:
            pending = selected - observed - protected
            if not pending:
                break
            static = roi_cost(conservative(selected))
            budget = deadline - time.monotonic()
            best, margin = None, 0
            # A completed passing probe gives evidence for the clean path unless a failure was seen.
            pass_chance = 0.75 if (observed and not failed) else 0.5
            # Filtered batches stop on failure. Put a worthwhile expensive
            # case first so a cheap failure cannot prevent resolving it.
            promoted = {
                n for n in pending & filtered_heavy
                if run_cost({n}) <= budget
                and pass_chance * (static - roi_cost(conservative(selected - {n}))) > run_cost({n})
            }
            seen = set()
            for batch in self.batches(pending, discounts):
                key = frozenset(batch)
                if not key or key in seen:
                    continue
                seen.add(key)
                estimate = run_cost(batch)
                if estimate > budget:
                    continue
                gain = static - roi_cost(conservative(selected - batch))
                if len(key) > 1 and (key & promoted or len(key) == 2 and key <= filtered_heavy):
                    first = min(batch, key=rank)
                    first_cost = run_cost({first})
                    first_gain = static - roi_cost(conservative(selected - {first}))
                    # A filtered batch stops on its first failure. If only its
                    # first method passes, that method still leaves the schedule.
                    net = (pass_chance * ((1 - pass_chance) * first_gain + pass_chance * gain)
                           - first_cost - pass_chance * (estimate - first_cost))
                elif failed or key & discounts.keys():
                    # A case can fail after earlier selected cases completed.
                    # Charge the filtered batch's full estimate in that event.
                    net = pass_chance * gain - estimate
                else:
                    first = min(batch, key=rank)
                    net = pass_chance * (gain - estimate) - (1 - pass_chance) * run_cost([first])
                if net > margin:
                    margin, best = net, batch

            replay = None
            for n in sorted((eligible & pending) - tried_replay, key=self.millis.get, reverse=True):
                gain = static - roi_cost(conservative(selected - {n}))
                if PROOF_CPU[n] < budget and 0.2 * gain - PROOF_CPU[n] > margin:
                    replay = n
                    break
            if replay is not None:
                n = replay
                tried_replay.add(n)
                start = cpu_time()
                spans, reason = proof.spans
                passed = False
                if spans:
                    if root is None:
                        root = get_session().apply_only(patch)
                    diagnostic = []
                    passed = can_prove_pass(n, spans, set(), root, deadline=deadline, diagnostic=diagnostic) is True
                    if diagnostic:
                        reason = diagnostic[0]
                log(f"proof {commit} n={n} passed={int(passed)} reason={reason} cpu={cpu_time()-start:.6f}")
                if passed:
                    selected.remove(n)
                continue

            if best is None:
                break
            log(f"decision {commit} batch={len(best)}/{len(pending)} est={run_cost(best):.3f} "
                f"gain={static-emitted_cost(conservative(selected-best)):.3f} early={int(not failed)} budget={budget:.2f}")
            root = None  # Session.run removes its overlay.
            if not run(best, not failed):
                break
        selected = conservative(selected)
        if subset is not None and self.memory in selected and self.memory not in observed:
            # An additional heap observer can enter only after another probe.
            # Price this optional follow-up against its marginal schedule gain.
            gain = emitted_cost(selected) - emitted_cost(selected - {self.memory})
            if .5 * gain > subset_cost:
                probe_memory()
        return selected
