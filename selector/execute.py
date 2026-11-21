"""Spend runner CPU only when passing tests can reduce the final worker bill."""

import ast
import os
import resource
import time
from collections import defaultdict

from runner.overlay import changes

from .analyze import numbers
from .proxies import can_prove_pass
from .scopes import snapshot
from .workers import schedule

CHEAP_MS = 1000 * float(os.getenv("SELECTOR_CHEAP_SECONDS", "0.5"))
CPU_RATIO = float(os.getenv("SELECTOR_CPU_RATIO", "1.393"))
IMPORT_CPU = float(os.getenv("SELECTOR_IMPORT_CPU", "0.188"))
# Increment over an ordinary class, measured with the method body omitted.
FIXTURE_CPU = {"tests/test_executor.py": 1.60, "tests/test_optimizer.py": 0.17}
PROOF_CPU = {1082: 0.2, 1106: 3.9, 1107: 0.3, 1123: 0.4, 1361: 1.3}


def cpu_time():
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time() + children.ru_utime + children.ru_stime


def proof_inputs(analyzer, patch):
    """Accept only replacements with unambiguous old statement coordinates."""
    spans, eligible = {}, 0
    for path, new in changes(analyzer.base, patch).items():
        if not (path.startswith("sqlglot/") and path.endswith(".py") and new):
            return None
        base = analyzer.base / path
        if not base.is_file():
            return None
        old = base.read_bytes()
        before, after = old.splitlines(), new.splitlines()
        if len(before) != len(after):
            return None
        funcs, scopes = analyzer.old(path)
        new_funcs, new_scopes = snapshot(ast.parse(new), new, funcs)
        if scopes != new_scopes or funcs.keys() != new_funcs.keys():
            return None
        intervals = []
        for name, (_, header, body, start, end) in funcs.items():
            if new_funcs[name][1] != header:
                return None
            if new_funcs[name][2] != body:
                intervals.append((start, end))
                eligible |= analyzer.index["by_func"].get(path, {}).get(name, 0)
        diff = [i for i, (a, b) in enumerate(zip(before, after), 1) if a != b]
        if not diff or not all(any(a <= i <= b for a, b in intervals) for i in diff):
            return None
        stmts = [(n.lineno, n.end_lineno) for n in ast.walk(ast.parse(old)) if isinstance(n, ast.stmt)]
        owners = set()
        for line in diff:
            enclosing = [(a, b) for a, b in stmts if a <= line <= b]
            if not enclosing:
                return None
            width = min(b - a for a, b in enclosing)
            owners.update((a, b) for a, b in enclosing if b - a == width)
        spans[path] = sorted(owners)
    return eligible, spans


class Execution:
    def __init__(self, analyzer, tests, millis):
        self.analyzer, self.millis = analyzer, millis
        self.ids = {row["number"]: row["id"] for row in tests}
        self.num = {node: n for n, node in self.ids.items()}
        self.files = {row["number"]: row["file"] for row in tests}
        self.classes = {n: node.rsplit("::", 1)[0] for n, node in self.ids.items()}
        profile = analyzer.index["durations"]
        # Process CPU omits spawned workers. Wall time bounds their contribution
        # in the single-CPU lane. Normalize against the measured full-suite ratio.
        weights = {int(n): max(cpu, wall - cpu) for n, (wall, cpu) in profile.items()}
        scale = CPU_RATIO * sum(millis.values()) / (1000 * sum(weights.values()))
        self.cpu = {n: scale * weights[n] for n in millis}
        # Unlike a warmed class in the duration map, the first submission
        # in each fresh optimizer class must spawn and import a worker.
        file = "tests/test_optimizer.py"
        hits = analyzer.line_hits(file)
        self.pool_users = set()
        for line, source in enumerate((analyzer.base / file).read_bytes().splitlines(), 1):
            if b"self.pool.submit(" in source:
                self.pool_users.update(numbers(hits.get(line, 0)))

    def run_cost(self, batch):
        fixtures = {self.files[n] for n in batch}
        setup = IMPORT_CPU + sum(FIXTURE_CPU.get(f, 0) for f in fixtures)
        if not self.pool_users.isdisjoint(batch):
            setup += 0.70
        return setup + sum(self.cpu[n] for n in batch)

    def ordered_batches(self, order):
        # Session groups by module/class. Split on a return to a previous
        # class or module so its actual fail-fast order matches our ranking.
        batch, files, classes = [], set(), set()
        last_file = last_class = None
        for n in order:
            f, cls = self.files[n], self.classes[n]
            if (f != last_file and f in files) or (cls != last_class and cls in classes):
                yield batch
                batch, files, classes = [], set(), set()
            batch.append(n)
            files.add(f)
            classes.add(cls)
            last_file, last_class = f, cls
        if batch:
            yield batch

    def cost(self, selected):
        return schedule(selected, self.millis)[1] / 1000

    def select(self, get_session, commit, patch, candidates, deadline, log):
        selected = set(numbers(candidates.mask))
        if not selected or get_session is None or candidates.broken:
            return selected
        failed = set()
        suite_observers = set()

        def run(order, early):
            budget = deadline - time.monotonic()
            if budget <= 0:
                return False
            start = cpu_time()
            result = get_session().run(patch, [self.ids[n] for n in order],
                                       stop_on_first_fail=early, overall_timeout=budget)
            used = cpu_time() - start
            bad = {self.num[node] for node in result.failing}
            failed.update(bad)
            if bad:
                # A fresh subset cannot reproduce allocations retained by an
                # earlier failure in the full suite. Keep process-wide heap
                # assertions when they import a changed module in that suite.
                for path in candidates.python_paths:
                    suite_observers.update(numbers(self.analyzer.index["heap_observers"] &
                                                   self.analyzer.importers[path]))
            if result.import_error:
                for path in candidates.python_paths:
                    selected.update(numbers(self.analyzer.importers[path] |
                                            self.analyzer.index["all_hits"].get(path, 0)))
            complete = "incomplete child run" not in result.notes
            # A killed process may still owe module/class teardown.
            if complete and not result.import_error:
                selected.difference_update(self.num[node] for node in result.ran if node not in result.failing)
            selected.update(suite_observers)
            log(f"run {commit} attempted={len(result.ran)}/{len(order)} failures={len(bad)} "
                f"cpu={used:.6f} child={result.cpu_s:.6f} complete={int(complete)}")
            return complete and not result.import_error

        def rank(n):
            bit = 1 << n
            return (0 if candidates.priority & bit else 1 if candidates.direct & bit else 2,
                    self.millis[n], n)

        cheap = {n for n in selected if self.millis[n] < CHEAP_MS}
        # A heavy test can pin the makespan. Count the option of reaching an
        # empty output only if every heavy test has an affordable removal path.
        heavy = selected - cheap
        fixed = {n for n in heavy if n not in PROOF_CPU}
        if len(fixed) == 1:
            n, = fixed
            if self.run_cost([n]) < self.cost(fixed):
                fixed.clear()
        if cheap:
            gain = self.cost(selected) - self.cost(heavy)
            if not fixed and heavy:
                gain += 2
            # The brief says most changes fail. Use even odds for the extra
            # cost of the last worker when the changed statements are reached.
            if not fixed and candidates.priority & candidates.direct:
                gain -= 1
            batches = list(self.ordered_batches(sorted(cheap, key=rank)))
            estimate = sum(self.run_cost(batch) for batch in batches)
            if estimate < gain:
                for batch in batches:
                    if not run(batch, True):
                        return selected
                    if failed:
                        break
            else:
                # Still inspect the first ranked class. A collection failure
                # can broaden the import closure even when removal saves zero.
                first = batches[0][0]
                probe = [first]
                body = self.cpu[first]
                for n in batches[0][1:]:
                    body += self.cpu[n]
                    if self.classes[n] != self.classes[first] or body > IMPORT_CPU:
                        break
                    probe.append(n)
                log(f"decision {commit} probe={len(probe)}/{len(cheap)} est={estimate:.3f} gain={gain:.3f}")
                if not run(probe, True):
                    return selected

        # Once failure is known, compare both the whole remainder and its
        # class-sized batches. Reprice after each batch using actual passes.
        if failed:
            pending = (cheap & selected) - failed - suite_observers
            while pending and time.monotonic() < deadline:
                groups = defaultdict(list)
                for n in sorted(pending, key=rank):
                    groups[self.classes[n]].append(n)
                best, margin = None, 0
                static = self.cost(selected)
                for batch in [list(pending), *groups.values()]:
                    estimate = self.run_cost(batch)
                    net = static - self.cost(selected.difference(batch)) - estimate
                    if net > margin and time.monotonic() + estimate < deadline:
                        best, margin = batch, net
                if best is None:
                    break
                if not run(sorted(best, key=rank), False):
                    return selected
                pending.difference_update(best)

        possible = (selected - failed - suite_observers) & heavy
        info = root = None
        checked = False
        for n in sorted(possible, key=lambda i: (-self.millis[i], i)):
            gain = self.cost(selected) - self.cost(selected - {n})
            dynamic = self.run_cost([n]) if len(selected) == 1 else float("inf")
            pass_fraction = 0.5 if candidates.priority & (1 << n) else 1.0
            proof = PROOF_CPU.get(n, float("inf"))
            if proof < min(gain, dynamic, deadline - time.monotonic()):
                start = cpu_time()
                if not checked:
                    info = proof_inputs(self.analyzer, patch)
                    checked = True
                proved = False
                if info and info[0] & (1 << n):
                    if root is None:
                        root = get_session().apply_only(patch)
                    proved = can_prove_pass(n, info[1], set(), root, deadline=deadline) is True
                log(f"proof {commit} n={n} passed={int(proved)} cpu={cpu_time()-start:.6f}")
                if proved:
                    selected.remove(n)
                    continue
            if dynamic < min(pass_fraction * gain, deadline - time.monotonic()):
                if not run([n], False):
                    return selected
        return selected
