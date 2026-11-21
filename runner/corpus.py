"""Collect full-suite results while another worker labels the growing local corpus."""

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from check import LIMITS, TRACK, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=TRACK / "lab/corpus/c1")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    data = args.data.resolve()
    out = TRACK / "scratch/runner/corpus" / data.name
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted((data / "commits").glob("*.patch"))
    commits = {p.stem: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    count = len(json.loads((data / "tests.json").read_bytes())["tests"])
    # A cached result is usable only for the same patch, runner implementation
    # and complete numbered catalog. Rebuilding the verification image is safe.
    runtime = hashlib.sha256()
    for name in ("__init__.py", "overlay.py", "cases.py", "process.py", "warm.py"):
        runtime.update((Path(__file__).parent / name).read_bytes())
    runtime.update((data / "tests.json").read_bytes())
    version = runtime.hexdigest()
    rows = {}
    for p in out.glob("*.json"):
        row = json.loads(p.read_bytes())
        if (p.stem in commits and row.get("sha256") == commits[p.stem] and row["ran"] == count
                and "incomplete child run" not in row["notes"]
                and row.get("runner") == version):
            rows[p.stem] = row

    def observe(commit):
        cmd = ["docker", "run", "--rm", *LIMITS, "-v", f"{data}:/data:ro",
               "br-runner", "python", "-m", "runner.verify", "--observe", "--commit", commit]
        row = json.loads(subprocess.check_output(cmd))
        if row["sha256"] != commits[commit] or row["ran"] != count:
            raise RuntimeError(f"{commit}: patch changed or suite interrupted")
        row["runner"] = version
        target = out / f"{commit}.json"
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(row), encoding="utf-8")
        tmp.replace(target)
        print(f"{commit}: {row['ran']} ran, {row['actual']} failing, {row['cpu']:.2f} CPU s", flush=True)
        return commit, row

    if not args.compare_only:
        pending = [commit for commit in commits if commit not in rows][:args.limit]
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            rows.update(pool.map(observe, pending))
    print(f"{len(rows)}/{len(commits)} complete observations", flush=True)
    answer_path = data / "answers.json"
    if not answer_path.is_file():
        print("Waiting for answers.json")
        return 2
    labels = json.loads(answer_path.read_bytes())["commits"]
    if set(labels) != set(commits):
        raise ValueError(f"patches and labels disagree in {data}")
    for commit, row in rows.items():
        expected = set(labels[commit]["failing"])
        found = set(row.pop("failing"))
        row.update(missing=sorted(expected - found), extra=sorted(found - expected))
    return int(bool(summarize(list(rows.values())) or len(rows) != len(commits)))


if __name__ == "__main__":
    raise SystemExit(main())
