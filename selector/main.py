"""Emit one scored record per patch from two independently guarded lanes."""

import argparse
import json
import math
import os
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

from runner import Session, cpu_time
from .analyze import Analyzer
from .execute import Execution
from .workers import schedule


def log(message):
    print(message, file=sys.stderr, flush=True)


def record(commit, workers):
    return json.dumps({"commit": commit, "workers": workers}, separators=(",", ":"))


def lane(execution, millis, patches, output, start, deadline, static, fallback):
    all_workers, all_cost = fallback
    with output.open("w", buffering=1) as dst, ExitStack() as stack:
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
                print(record(path.stem, all_workers), file=dst)
                continue
            # Reserve analysis time for every remaining patch. Allow unused
            # slots to accumulate after empty/cheap commits.
            remaining = len(patches) - position
            run_until = min(deadline - 0.25 * remaining - 0.5, start + slot * (position + 1))
            guarded = static or now >= run_until
            begin = cpu_time()
            try:
                patch = path.read_bytes().decode("utf-8", "surrogateescape")
                candidates = execution.analyzer.analyze(patch, prioritize=not guarded,
                                                        session=session, deadline=deadline)
                selected = execution.select(None if guarded else session, path.stem, patch,
                                            candidates, run_until, log)
                workers, cost = schedule(selected, millis)
                # For a broad static selection the largest indivisible test
                # can already set the full-catalog price.
                if cost >= all_cost and not candidates.structural:
                    selected.update(n for n in millis if not candidates.mask & (1 << n))
                    workers = [[n for n in w if n in selected] for w in all_workers]
                    workers = [w for w in workers if w]
                    cost = max(sum(millis[n] for n in w) for w in workers) + 2000 * len(workers)
                log(f"commit {path.stem} candidates={candidates.mask.bit_count()} kept={len(selected)} "
                    f"sim={cost/1000:.3f} cpu={cpu_time()-begin:.6f} guard={int(guarded)} "
                    f"broken={int(candidates.broken)}")
            except Exception as exc:
                log(f"fallback {path.stem} {exc!r}")
                workers = all_workers
            print(record(path.stem, workers), file=dst)


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
    # A 30-commit practice batch has 180 s, a final batch about 45 has 270 s.
    # Keep the shorter cap for ambiguous intermediate sizes.
    limit = 270 if len(paths) >= 45 else 180
    deadline = start + min(6 * len(paths), limit) - 3
    try:
        mark = time.process_time()
        analyzer = Analyzer(manifest)
    except Exception as exc:
        log(f"fallback analyzer {exc!r}")
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
            if paths and not static:
                warm()
        except Exception as exc:
            log(f"fallback startup {exc!r}")
            execution = None
    if execution is None:
        for path in paths:
            print(record(path.stem, fallback[0]))
        paths = []
    log(f"startup_cpu={time.process_time():.6f}")
    with tempfile.TemporaryDirectory() as tmp:
        jobs = []
        for index in range(min(2, len(paths))):
            output = Path(tmp) / str(index)
            batch = paths[index::2]
            pid = os.fork()
            if not pid:
                os.dup2(2, 1)
                try:
                    lane(execution, millis, batch, output, start, deadline, static, fallback)
                except BaseException as exc:
                    log(f"lane {index} {exc!r}")
                    os._exit(1)
                os._exit(0)
            jobs.append((pid, batch, output))
        for pid, batch, output in jobs:
            os.waitpid(pid, 0)
            emitted = set()
            if output.exists():
                for line in output.read_text().splitlines():
                    try:
                        emitted.add(json.loads(line)["commit"])
                    except (ValueError, KeyError):
                        continue
                    print(line)
            for path in batch:
                if path.stem not in emitted:
                    log(f"fallback {path.stem} incomplete lane")
                    print(record(path.stem, fallback[0]))
    sys.stdout.flush()
    usage = dict(line.split() for line in Path("/sys/fs/cgroup/cpu.stat").read_text().splitlines())
    log(f"CPU_USEC={usage['usage_usec']}")


if __name__ == "__main__":
    main()
