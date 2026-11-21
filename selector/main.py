"""Emit a single scorer record for every patch, isolating each lane's stdout."""

import argparse
import json
import os
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

from runner import Session
from .analyze import Analyzer
from .execute import Execution
from .workers import schedule


def log(message):
    print(message, file=sys.stderr, flush=True)


def lane(analyzer, millis, patches, output, start, deadline, static):
    execution = Execution(analyzer, millis)
    # Each lane owns its overlay. Session retains the pristine imports across
    # fixture edits and performs its one-time warmup on first use.
    session = nullcontext(None) if static else Session()
    with output.open("w", buffering=1) as dst, session as runner:
        for position, path in enumerate(patches):
            now = time.monotonic()
            slot = (deadline - start) / len(patches)
            if position and not static and now > start + slot * position:
                static = True
                log(f"guard {path.stem} lane behind schedule; remaining={len(patches)-position} static")
            try:
                patch = path.read_bytes().decode("utf-8", "surrogateescape")
                candidates = analyzer.analyze(patch, prioritize=not static)
                # Leave time to analyze and emit every remaining patch statically.
                reserve = 0.25 * (len(patches) - position) + 0.5
                run_until = min(deadline - reserve, start + slot * (position + 1), now + slot)
                selected = execution.select(None if static else runner, path.stem, patch, candidates,
                                            run_until, log)
                workers, _ = schedule(selected, millis)
            except Exception as e:
                log(f"fallback {path.stem} {e!r}")
                workers, _ = schedule(analyzer.ids, millis, forced=8)
            print(json.dumps({"commit": path.stem, "workers": workers}, separators=(",", ":")), file=dst)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("/data"))
    p.add_argument("--static", action="store_true")
    args = p.parse_args()
    wall = time.monotonic()
    start = time.process_time()
    analyzer = Analyzer(args.data / "tests.json")
    log(f"map_load_cpu={time.process_time()-start:.4f}")
    millis = {t["number"]: round(t["seconds"] * 1000) for t in analyzer.tests}
    paths = sorted((args.data / "commits").glob("*.patch"))
    deadline = wall + 5 * len(paths)
    static = args.static or os.environ.get("SELECTOR_STATIC") == "1"
    if paths and not static:
        # Executor/optimizer test modules import these locked dependencies. Warm
        # them once before both forks, without importing any patchable modules.
        # Redirect native writes too: only the parent emits records on stdout.
        stdout = os.dup(1)
        try:
            os.dup2(2, 1)
            import duckdb
            import pandas.testing
        finally:
            sys.stdout.flush()
            os.dup2(stdout, 1)
            os.close(stdout)
    with tempfile.TemporaryDirectory() as tmp:
        jobs = []
        for index in range(min(2, len(paths))):
            output = Path(tmp) / str(index)
            batch = paths[index::2]
            pid = os.fork()
            if not pid:
                os.dup2(2, 1)
                try:
                    lane(analyzer, millis, batch, output, wall, deadline, static)
                except BaseException as e:
                    log(f"lane {index} {e!r}")
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
                    except ValueError:
                        continue
                    print(line)
            for path in batch:
                if path.stem not in emitted:
                    log(f"fallback {path.stem} incomplete lane")
                    workers, _ = schedule(analyzer.ids, millis, forced=8)
                    print(json.dumps({"commit": path.stem, "workers": workers}, separators=(",", ":")))
    sys.stdout.flush()
    usage = dict(line.split() for line in Path("/sys/fs/cgroup/cpu.stat").read_text().splitlines())
    log(f"CPU_USEC={usage['usage_usec']}")


if __name__ == "__main__":
    main()
