"""Quickly replay dropped test methods before labeling entire mutant suites."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory

from label import read_report
from unidiff import apply_patch


def check(patch, tests):
    with TemporaryDirectory() as tmp:
        tree, report = Path(tmp, "sqlglot"), Path(tmp, "report.xml")
        shutil.copytree("/sqlglot", tree, ignore=shutil.ignore_patterns(".git", "__pycache__"))
        apply_patch(tree, patch.read_bytes())
        command = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "--timeout=60",
                   f"--junitxml={report}", "-q", *[test["id"] for test in tests]]
        run = subprocess.run(command, cwd=tree, env=dict(os.environ, PYTHONPATH=str(tree)),
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
        if run.returncode not in (0, 1, 5) or not report.exists():
            raise RuntimeError(f"{patch.name}: {run.stdout[-3000:].decode(errors='replace')}")
        failures = read_report(report, {"tests": tests})
        if failures:
            print(f"{patch.name}: {run.stdout[-3000:].decode(errors='replace')}", file=sys.stderr, flush=True)
        return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    catalog = {row["number"]: row for row in json.loads(args.manifest.read_bytes())["tests"]}
    drops = json.loads((args.labels / "drops.json").read_bytes())
    total = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pending = {pool.submit(check, args.labels / "commits" / f"{commit}.patch",
                               [catalog[n] for n in numbers]): commit for commit, numbers in drops.items()}
        for job in as_completed(pending):
            commit = pending[job]
            failures = job.result()
            print(json.dumps({"commit": commit, "dropped": len(drops[commit]), "failing": failures}), flush=True)
            total += len(failures)
    print(f"{len(drops)} mutants, {sum(map(len, drops.values()))} dropped pairs, {total} failed", flush=True)
    raise SystemExit(bool(total))


if __name__ == "__main__":
    main()
