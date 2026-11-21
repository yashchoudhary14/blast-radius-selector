"""Run the offline warm/cold comparison under the grader's container limits."""
import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from check import LIMITS, TRACK


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="br-o")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--commit", action="append", default=[])
    args = parser.parse_args()
    out = args.out or TRACK / "scratch" / "o-warm" / ("retry" if args.commit else "gate")
    out.mkdir(parents=True, exist_ok=True)
    sets = [TRACK / "sample/sample", TRACK / "lab/corpus/c1", TRACK / "lab/corpus/c2"]
    mounts = [item for path in sets for item in ("-v", f"{path}:/sets/{path.name}:ro")]
    base = ["docker", "run", "--rm", *LIMITS, *mounts,
            args.image, "python", "-m", "runner.verify_warm", *[f"/sets/{d.name}" for d in sets]]
    extra = [item for name in args.commit for item in ("--commit", name)]

    def shard(i):
        with (out / f"{i}.jsonl").open("w") as sink:
            return subprocess.run([*base, "--shard", f"{i}/{args.jobs}", *extra], stdout=sink).returncode

    with ThreadPoolExecutor(args.jobs) as pool:
        status = list(pool.map(shard, range(args.jobs)))
    rows = [json.loads(line) for i in range(args.jobs) for line in (out / f"{i}.jsonl").read_bytes().splitlines()]
    for data in sets:
        group = [r for r in rows for name in r.get("commits", []) if name.startswith(data.name + "/")]
        print(data.name, len(group), "commits,", sum(r["selected"] for r in group), "tests,",
              sum(len(r["diff"]) for r in group), "mismatches,", sum(not r["complete"] for r in group), "incomplete")
    return int(any(status))


if __name__ == "__main__":
    raise SystemExit(main())
