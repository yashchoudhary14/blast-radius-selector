"""Compare complete static candidate lists and stage only fresh mutants with drops."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
WORK = ROOT / "scratch/r"
DATA = [ROOT / "sample/sample", *(ROOT / "lab/corpus" / name for name in ("c1", "c2", "c3", "c3b", "adv")),
        WORK / "fresh"]


def check(image, data):
    out = WORK / f"{image}-{data.name}.jsonl"
    err = out.with_suffix(".log")
    labeled = (data / "answers.json").is_file()
    cmd = ["docker", "run", "--rm", "-v", f"{data}:/data:ro", image, "python", "-m",
           "selector.analyze", "/data", *(["--check"] if labeled else [])]
    with out.open("wb") as stdout, err.open("wb") as stderr:
        result = subprocess.run(cmd, stdout=stdout, stderr=stderr)
    diag = err.read_text(encoding="utf8")
    if result.returncode not in (0, 1) or "fallback " in diag or (labeled and "recall=" not in diag):
        raise RuntimeError(f"{cmd}: {diag[-500:]}")
    rows = load(out)
    assert set(rows) == {f.stem for f in (data / "commits").glob("*.patch")}
    if labeled:
        subprocess.run([sys.executable, "lab/eval/misses.py", str(out), str(data)], check=True)
    print(image, data.name, sum(map(len, rows.values())), diag.splitlines()[-1], flush=True)
    return rows


def load(path):
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    return {row["commit"]: set(row["candidates"]) for row in rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--base")
    p.add_argument("--set", nargs="*")
    args = p.parse_args()
    for data in DATA:
        if args.set and data.name not in args.set:
            continue
        after = check(args.image, data)
        if not args.base:
            continue
        before = load(WORK / f"{args.base}-{data.name}.jsonl")
        dropped = {c: sorted(before[c] - row) for c, row in after.items() if before[c] - row}
        print(f"{len(dropped)} narrowed commits, {sum(map(len,dropped.values()))} dropped pairs", flush=True)
        if data.name == "fresh":
            selected = WORK / "labels/commits"
            selected.mkdir(exist_ok=True, parents=True)
            for c in dropped:
                shutil.copyfile(data / "commits" / (c + ".patch"), selected / (c + ".patch"))
            assert {p.stem for p in selected.glob("*.patch")} == set(dropped)
            (selected.parent / "drops.json").write_text(json.dumps(dropped))
            continue
        answers = json.loads((data / "answers.json").read_bytes())["commits"]
        new_misses = {c: sorted(set(row) & set(answers[c]["failing"])) for c, row in dropped.items()}
        print("new missed failures", {c: v for c, v in new_misses.items() if v}, flush=True)


if __name__ == "__main__":
    main()
