"""Emit one scored record per patch from two independently guarded lanes."""

import argparse
import json
import math
import os
import select
import signal
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from runner import Session, cpu_time
from runner.overlay import PatchError
from .analyze import Analyzer
from .execute import Execution
from .workers import WORKER_MS, schedule


def log(message):
    print(message, file=sys.stderr, flush=True)


def record(commit, workers):
    return json.dumps({"commit": commit, "workers": workers}, separators=(",", ":"))


def log_price(commit, cost, detail):
    log("full-price " + json.dumps({"commit": commit, "cost_ms": cost, **detail}, separators=(",", ":")))


def selection_price(candidates, selected, millis, cost, expanded):
    heavy = []
    for n in sorted(selected, key=lambda n: (-millis[n], n))[:8]:
        bit = 1 << n
        heavy.append(dict(test=n, ms=millis[n], without_ms=schedule(selected - {n}, millis)[1],
                          direct=bool(candidates.direct & bit), broad=bool(candidates.broad & bit)))
    return dict(reason="catalog-schedule" if expanded else "selection-schedule", paths=candidates.paths,
                rules=candidates.reasons, selected=len(selected), selected_ms=cost, heavy=heavy)


def lane(execution, millis, patches, dst, start, deadline, static, fallback, *, audited_pipe=False):
    all_workers, all_cost = fallback

    def emit(commit, workers, audit):
        if audit is not None:
            if audited_pipe:
                print(json.dumps(dict(commit=commit, workers=workers, audit=audit), separators=(",", ":")), file=dst)
                return
            log_price(commit, all_cost, audit)
        print(record(commit, workers), file=dst)

    with ExitStack() as stack:
        runner = None

        def session():
            nonlocal runner
            if runner is None:
                runner = stack.enter_context(Session())
            return runner

        slot = (deadline - start) / len(patches)
        for position, path in enumerate(patches):
            now = time.monotonic()
            if now >= deadline:
                log(f"fallback {path.stem} lane deadline")
                emit(path.stem, all_workers, dict(reason="lane-deadline"))
                continue
            # Reserve analysis time for every remaining patch. Allow unused
            # slots to accumulate after empty/cheap commits.
            remaining = len(patches) - position
            run_until = min(deadline - 0.25 * remaining - 0.5, start + slot * (position + 1))
            guarded = static or now >= run_until
            begin = cpu_time()
            audit = None
            stage = "patch-read"
            try:
                patch = path.read_bytes().decode("utf-8", "surrogateescape")
                stage = "analysis"
                candidates = execution.analyzer.analyze(
                    patch,
                    prioritize=not guarded,
                    session=None if guarded else session,
                    deadline=deadline,
                    run_until=None if static else run_until,
                    allow_probe=not guarded,
                )
                stage = "selection"
                selected = execution.select(None if guarded else session, path.stem, patch,
                                            candidates, run_until, log)
                stage = "schedule"
                workers, cost = schedule(selected, millis)
                selected_cost, pre_expansion = cost, selected
                # For a broad static selection the largest indivisible test
                # can already set the full-catalog price.
                expanded = cost >= all_cost and not candidates.structural
                if expanded:
                    selected = selected | {n for n in millis if not candidates.mask & (1 << n)}
                    workers = [[n for n in w if n in selected] for w in all_workers]
                    workers = [w for w in workers if w]
                    cost = max(sum(millis[n] for n in w) for w in workers) + WORKER_MS * len(workers)
                if cost == all_cost:
                    audit = selection_price(candidates, pre_expansion, millis, selected_cost, expanded)
                log(f"commit {path.stem} candidates={candidates.mask.bit_count()} kept={len(selected)} "
                    f"sim={cost/1000:.3f} cpu={cpu_time()-begin:.6f} guard={int(guarded)} "
                    f"broken={int(candidates.broken)}")
            except Exception as exc:
                log(f"fallback {path.stem} {exc!r}")
                workers = all_workers
                audit = dict(reason=exc.reason if isinstance(exc, PatchError) else "exception",
                             stage=stage, error=repr(exc))
            emit(path.stem, workers, audit)


def warm():
    # These locked dependencies are used by both executor/optimizer lanes.
    # Keep patched SQLGlot and test imports local to each runner child.
    stdout = os.dup(1)
    try:
        os.dup2(2, 1)
        import duckdb
        import pandas.testing
    finally:
        sys.stdout.flush()
        os.dup2(stdout, 1)
        os.close(stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/data"))
    parser.add_argument("--static", action="store_true")
    args = parser.parse_args()
    start = time.monotonic()
    paths = sorted((args.data / "commits").glob("*.patch"))
    manifest = args.data / "tests.json"
    static = args.static or os.getenv("SELECTOR_STATIC") == "1"
    # A 30-commit practice batch has 180 s, a final batch (~45-60) has 270 s.
    # Larger local evaluation batches (e.g. c2 with 300 commits) run under 180 s.
    limit = 270 if 45 <= len(paths) <= 120 else 180
    margin = max(8.0, 0.04 * limit)
    hard_deadline = start + limit - margin
    deadline = min(start + max(10.0, 6.0 * len(paths)) - 3.0, hard_deadline - 2.0)
    startup_audit = None
    try:
        mark = time.process_time()
        analyzer = Analyzer(manifest)
    except Exception as exc:
        log(f"fallback analyzer {exc!r}")
        startup_audit = dict(reason="analyzer-startup", error=repr(exc))
        analyzer = None
    tests = getattr(analyzer, "tests", None)
    if tests is None:
        tests = json.loads(manifest.read_bytes())["tests"]
    millis = {row["number"]: math.floor(row["seconds"] * 1000 + 0.5) for row in tests}
    fallback = schedule(millis, millis)
    execution = None
    if analyzer is not None:
        try:
            execution = Execution(analyzer, tests, millis)
            log(f"map_load_cpu={time.process_time()-mark:.6f}")
            if paths and not static and time.monotonic() < deadline:
                warm()
        except Exception as exc:
            log(f"fallback startup {exc!r}")
            startup_audit = dict(reason="execution-startup", error=repr(exc))
            execution = None
    if execution is None or time.monotonic() >= hard_deadline:
        if execution is not None:
            log("fallback startup deadline")
            startup_audit = dict(reason="startup-deadline")
        for path in paths:
            log_price(path.stem, fallback[1], startup_audit)
            print(record(path.stem, fallback[0]))
        paths = []
    log(f"startup_cpu={time.process_time():.6f}")
    if paths:
        active = {}
        emitted = set()
        expected = {path.stem for path in paths}
        for index in range(min(2, len(paths))):
            rd, wr = os.pipe()
            batch = paths[index::2]
            pid = os.fork()
            if not pid:
                os.setpgid(0, 0)
                os.close(rd)
                for other_rd in active:
                    os.close(other_rd)
                os.dup2(2, 1)
                try:
                    with os.fdopen(wr, "w", buffering=1, encoding="utf-8") as dst:
                        lane(execution, millis, batch, dst, start, deadline, static, fallback, audited_pipe=True)
                except BaseException as exc:
                    log(f"lane {index} {exc!r}")
                    os._exit(1)
                os._exit(0)
            try:
                os.setpgid(pid, pid)
            except OSError:
                pass
            os.close(wr)
            active[rd] = [pid, bytearray()]
        while active and time.monotonic() < hard_deadline:
            timeout = max(0.0, hard_deadline - time.monotonic())
            ready, _, _ = select.select(list(active), [], [], timeout)
            if not ready:
                break
            for fd in ready:
                chunk = os.read(fd, 65536)
                pid, buf = active[fd]
                if not chunk:
                    os.close(fd)
                    del active[fd]
                    os.waitpid(pid, 0)
                    continue
                buf.extend(chunk)
                *lines, tail = buf.split(b"\n")
                active[fd][1] = bytearray(tail)
                for raw in lines:
                    try:
                        text = raw.decode("utf-8")
                        row = json.loads(text)
                        commit = row["commit"]
                    except (UnicodeDecodeError, ValueError, KeyError, TypeError):
                        continue
                    if commit in expected and commit not in emitted:
                        emitted.add(commit)
                        if "audit" in row:
                            log_price(commit, fallback[1], row["audit"])
                            text = record(commit, row["workers"])
                        print(text, flush=True)
        missing_reason = "watchdog-deadline" if active else "incomplete-lane"
        if active:
            missing = len(paths) - len(emitted)
            log(f"watchdog deadline limit={limit} missing={missing}")
            for fd, (pid, _) in list(active.items()):
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                os.close(fd)
                os.waitpid(pid, 0)
        for path in paths:
            if path.stem not in emitted:
                emitted.add(path.stem)
                log(f"fallback {path.stem} incomplete lane")
                log_price(path.stem, fallback[1], dict(reason=missing_reason))
                print(record(path.stem, fallback[0]))
    sys.stdout.flush()
    stat_path = Path("/sys/fs/cgroup/cpu.stat")
    if stat_path.is_file():
        usage = dict(line.split() for line in stat_path.read_text().splitlines())
        log(f"CPU_USEC={usage['usage_usec']}")


if __name__ == "__main__":
    main()
