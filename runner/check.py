"""Build br-runner and reproduce the label gate and CPU measurements from the host."""

import argparse
import json
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


HERE = Path(__file__).resolve().parent
TRACK = HERE.parents[1]
LIMITS = [
    "--network", "none", "--cpus", "2", "--memory", "4096m", "--read-only",
    "--tmpfs", "/tmp:rw,size=64m,mode=1777", "--user", "65534:65534",
    "--env", "HOME=/tmp", "--pids-limit", "256",
]


def summarize(rows):
    failures = 0
    for size in sorted({r["selected"] for r in rows}):
        group = [r for r in rows if r["selected"] == size]
        bad = [r for r in group if r["missing"] or r["extra"] or r["ran"] != size
               or "incomplete child run" in r["notes"]]
        cpu = [r["cpu"] for r in group]
        recorded = sum(r["recorded"] for r in group)
        print(f"{size} tests | exact {len(group)-len(bad)}/{len(group)} | "
              f"child CPU mean {statistics.mean(cpu):.3f}, median {statistics.median(cpu):.3f}, "
              f"range {min(cpu):.3f}..{max(cpu):.3f} s | ratio {sum(cpu)/recorded:.3f} | "
              f"mean parent {statistics.mean(r['parent_cpu'] for r in group):.3f} s")
        for row in bad:
            print(json.dumps(row))
        failures += len(bad)
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--data", type=Path, action="append", help="labeled dataset (repeatable)")
    parser.add_argument("--report", type=Path, nargs="+", help="summarize existing JSONL files")
    args = parser.parse_args()
    if args.report:
        rows = [json.loads(line) for p in args.report
                for line in p.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        return int(bool(summarize(rows)))
    subprocess.run(["docker", "build", "--network", "none", "-t", "br-runner",
                    "-f", str(HERE / "Dockerfile.test"), str(HERE)], check=True)
    run = ["docker", "run", "--rm", *LIMITS]
    for module in ("test_runner", "benchmark"):
        subprocess.run([*run, "-v", f"{TRACK/'sample/sample'}:/data:ro",
                        "br-runner", "python", "-m", f"runner.{module}"], check=True)
    datasets = args.data or [TRACK / "sample/sample"]
    c1 = TRACK / "lab/corpus/c1"
    status = 0
    for data in datasets:
        data = data.resolve()
        out = TRACK / "scratch/runner" / data.name
        out.mkdir(parents=True, exist_ok=True)
        expected = {p.stem for p in (data / "commits").glob("*.patch")}
        if not expected or expected != set(json.loads((data / "answers.json").read_bytes())["commits"]):
            raise ValueError(f"incomplete labels in {data}")
        jobs = min(args.jobs, len(expected))

        def shard(index):
            path = out / f"{index}.jsonl"
            with path.open("w", encoding="utf-8") as sink:
                cmd = [*run, "-v", f"{data}:/data:ro", "br-runner",
                       "python", "-m", "runner.verify", "--shard", f"{index}/{jobs}"]
                proc = subprocess.run(cmd, stdout=sink)
            return proc.returncode, [json.loads(line) for line in path.read_bytes().splitlines()]

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(shard, range(jobs)))
        rows = [row for _, group in results for row in group]
        print(data)
        status |= bool(summarize(rows) or any(code for code, _ in results))
        for size in (50, len(json.loads((data / "tests.json").read_bytes())["tests"])):
            status |= {r["commit"] for r in rows if r["selected"] == size} != expected
    if not args.data and (c1 / "tests.json").is_file():
        corpus = subprocess.run([sys.executable, "-B", str(HERE / "corpus.py"),
                                 "--data", str(c1), "--jobs", str(args.jobs)])
        status |= bool(corpus.returncode)
    return int(status)


if __name__ == "__main__":
    raise SystemExit(main())
