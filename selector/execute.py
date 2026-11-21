"""Choose between executing the remaining cheap tests and paying their worker loads."""

import os
import time

from .analyze import numbers
from .workers import schedule

# The full-catalog runner benchmark measured 1.393 CPU seconds per recorded
# second. Use baseline per-method CPU measurements to distribute that estimate.
CPU_RATIO = float(os.environ.get("SELECTOR_CPU_RATIO", "1.393"))
IMPORT_CPU = float(os.environ.get("SELECTOR_IMPORT_CPU", "0.188"))


class Execution:
    def __init__(self, analyzer, millis):
        self.analyzer = analyzer
        self.millis = millis
        profile = analyzer.index["durations"]
        total = sum(v[1] for v in profile.values())
        scale = CPU_RATIO * sum(millis.values()) / (1000 * total)
        self.cpu = {int(n): pair[1] * scale for n, pair in profile.items()}

    def select(self, session, commit, patch, candidates, deadline, log):
        analyzer = self.analyzer
        selected = set(numbers(candidates.mask))
        if session is None or candidates.broken:
            return selected
        def rank(n):
            bit = 1 << n
            tier = 0 if candidates.priority & bit else 1 if candidates.direct & bit else 2
            return tier, self.millis[n], n

        cheap = sorted((n for n in selected if self.millis[n] < 500), key=rank)
        if not cheap:
            return selected

        def run(order, early):
            budget = deadline - time.monotonic()
            if budget <= 0:
                return set(), set(), True
            result = session.run(patch, [analyzer.ids[n] for n in order],
                                 stop_on_first_fail=early, overall_timeout=budget)
            failing = {analyzer.num[i] for i in result.failing}
            passed = {analyzer.num[i] for i in result.ran} - failing
            interrupted = "incomplete child run" in result.notes
            log(f"run {commit} attempted={len(result.ran)}/{len(order)} failing={sorted(failing)} "
                f"cpu={result.cpu_s:.3f} notes={result.notes[:2]}")
            if interrupted:
                log(f"guard {commit} unresolved={len(order)-len(result.ran)}")
            if result.import_error:
                # A valid module can raise during import. Other test files can
                # fail collection even if they never executed the changed code.
                for path in candidates.python_paths:
                    selected.update(numbers(analyzer.importers[path] | analyzer.index["all_hits"].get(path, 0)))
            return passed, failing, interrupted

        passed, failing, interrupted = run(cheap, True)
        selected -= passed
        remaining = [n for n in cheap if n in selected and n not in failing]
        if not failing or not remaining or interrupted:
            return selected
        _, static_cost = schedule(selected, self.millis)
        _, floor = schedule(selected.difference(remaining), self.millis)
        cost = IMPORT_CPU + sum(self.cpu[n] for n in remaining)
        saving = (static_cost - floor) / 1000
        keep_running = cost < saving and time.monotonic() + cost < deadline
        log(f"decision {commit} {'run' if keep_running else 'static'} remaining={len(remaining)} "
            f"est_cpu={cost:.3f} max_saving={saving:.3f}")
        if keep_running:
            passed, _, _ = run(remaining, False)
            selected -= passed
        return selected
