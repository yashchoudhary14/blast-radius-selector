"""Compare measured runner CPU with the marginal cost of the emitted worker schedule."""

import resource
import time
from collections import defaultdict

from .analyze import numbers
from .patch_proof import PatchProof
from .proxies import can_prove_pass
from .workers import schedule


# Two busy lanes under the scorer quota. Fit the per-module charge on ordinary batches, then
# regress child CPU on recorded seconds and fixture presence across all batch sizes.
STARTUP = 0.084
PER_SECOND = 1.001
PER_MODULE = 0.0132
FIXTURE = {"tests/test_executor.py": 1.173, "tests/test_optimizer.py": 0.534}
# Pristine single-test child CPU under the same limits, inclusive of collection.
SINGLE = {
    773: .655, 1082: 1.645, 1106: 33.073, 1107: 2.367, 1123: 5.487,
    1224: 2.198, 1295: 1.895, 1313: 1.069, 1343: 2.041, 1358: .890,
    1361: 1.762, 1382: 3.069, 1415: 1.100, 1428: 2.271, 1434: 1.937,
    1440: 6.215, 1490: 29.103, 1541: .301, 1560: 1.024, 1561: .442,
}
PROOF_CPU = {1082: .3, 1106: 4.5, 1107: .4, 1123: .5, 1361: 1.4}


def cpu_time():
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time() + children.ru_utime + children.ru_stime


class Execution:
    def __init__(self, analyzer, tests, millis):
        self.analyzer, self.millis = analyzer, millis
        self.ids = {t["number"]: t["id"] for t in tests}
        self.num = {node: n for n, node in self.ids.items()}
        self.files = {t["number"]: t["file"] for t in tests}
        self.body = {n: PER_SECOND * ms / 1000 for n, ms in millis.items()}
        for n, total in SINGLE.items():
            if n in self.body:
                self.body[n] = max(0, total - self.setup({self.files[n]}))

    def setup(self, files):
        return STARTUP + PER_MODULE * len(files) + sum(FIXTURE.get(f, 0) for f in files)

    def run_cost(self, batch):
        return self.setup({self.files[n] for n in batch}) + sum(self.body[n] for n in batch)

    def cost(self, selected):
        return schedule(selected, self.millis)[1] / 1000

    def batches(self, pending):
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
        for n in sorted(heavy):
            yield {n}
        yield from groups.values()

    def select(self, get_session, commit, patch, candidates, deadline, log):
        selected = set(numbers(candidates.mask))
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
            if remaining - potential_observers:
                return remaining | potential_observers
            return remaining

        if get_session is None:
            return conservative(selected)
        proof = root = None
        tried_replay = set()

        def rank(n):
            level = 0 if candidates.priority & (1 << n) else 1 if candidates.direct & (1 << n) else 2
            return level, self.run_cost([n]), n

        def run(batch, early):
            order = sorted(batch, key=rank)
            start = cpu_time()
            budget = deadline - time.monotonic()
            if budget <= 0:
                return False
            result = get_session().run(patch, [self.ids[n] for n in order],
                                       stop_on_first_fail=early, overall_timeout=budget)
            bad = {self.num[node] for node in result.failing}
            failed.update(bad)
            if bad:
                protected.update(potential_observers)
            if result.import_error:
                for path in candidates.python_paths:
                    selected.update(numbers(self.analyzer.importers[path] | self.analyzer.index["all_hits"].get(path, 0)))
            complete = "incomplete child run" not in result.notes
            ran = {self.num[node] for node in result.ran}
            if complete and not result.import_error:
                observed.update(ran)
                selected.difference_update(ran - bad)
            selected.update(protected)
            log(f"run {commit} attempted={len(ran)}/{len(order)} failures={len(bad)} cpu={cpu_time()-start:.6f} "
                f"child={result.cpu_s:.6f} recorded={sum(self.millis[n] for n in ran)/1000:.3f} "
                f"modules={len({self.files[n] for n in ran})} complete={int(complete)}")
            return complete and not result.import_error and bool(ran)

        eligible = set(PROOF_CPU) & selected
        if eligible:
            proof = PatchProof(self.analyzer, patch)
            for n in sorted(eligible, key=self.millis.get, reverse=True):
                start = cpu_time()
                reason = proof.witness(n)
                if reason:
                    selected.remove(n)
                log(f"proof {commit} n={n} passed={int(bool(reason))} reason={reason or 'no-witness'} "
                    f"cpu={cpu_time()-start:.6f}")

        probed = False
        while time.monotonic() < deadline:
            pending = selected - observed - protected
            if not pending:
                break
            static = self.cost(conservative(selected))
            budget = deadline - time.monotonic()
            best, margin = None, 0
            # A completed passing probe gives evidence for the clean path.
            pass_chance = 0.75 if observed else 0.5
            seen = set()
            for batch in self.batches(pending):
                key = frozenset(batch)
                if not key or key in seen:
                    continue
                seen.add(key)
                estimate = self.run_cost(batch)
                if estimate > budget:
                    continue
                gain = static - self.cost(conservative(selected - batch))
                if failed:
                    net = gain - estimate
                else:
                    first = min(batch, key=rank)
                    net = pass_chance * (gain - estimate) - (1 - pass_chance) * self.run_cost([first])
                if net > margin:
                    margin, best = net, batch

            replay = None
            for n in sorted((eligible & pending) - tried_replay, key=self.millis.get, reverse=True):
                gain = static - self.cost(conservative(selected - {n}))
                if PROOF_CPU[n] < budget and gain - PROOF_CPU[n] > margin:
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

            cheap = {n for n in pending if self.millis[n] < 500}
            if best is None:
                if probed or not cheap or not candidates.python_paths:
                    break
                # Resolve import failures even when an indivisible heavy test
                # pins the price. A broad import error changes the candidate set.
                best = {min(cheap, key=rank)}
                if self.run_cost(best) > budget:
                    break
            log(f"decision {commit} batch={len(best)}/{len(pending)} est={self.run_cost(best):.3f} "
                f"gain={static-self.cost(conservative(selected-best)):.3f} early={int(not failed)} budget={budget:.2f}")
            probed = True
            root = None  # Session.run removes its overlay.
            if not run(best, not failed):
                break
        return conservative(selected)
