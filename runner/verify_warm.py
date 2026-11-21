"""Offline comparison of patched fork snapshots with the cold reference."""
import argparse
import hashlib
import json
import random
from pathlib import Path

from runner import GENERATOR_MEMORY, Session
from selector.analyze import Analyzer, numbers
from selector.main import warm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", type=Path, nargs="+")
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--commit", action="append")
    args = parser.parse_args()
    index, shards = map(int, args.shard.split("/"))
    patches = {}
    for dataset in args.datasets:
        manifest = dataset / "tests.json"
        tests = json.loads(manifest.read_bytes())["tests"]
        labels = json.loads((dataset / "answers.json").read_bytes())["commits"]
        for commit, row in labels.items():
            if args.commit and commit not in args.commit:
                continue
            data = (dataset / "commits" / (commit + ".patch")).read_bytes()
            digest = hashlib.sha256(data).digest()
            patch, failing, names = patches.setdefault(digest, (data, set(), []))
            failing.update(row["failing"])
            names.append(f"{dataset.name}/{commit}")
    analyzer = Analyzer(manifest)
    catalog = {row["number"]: row for row in tests}
    budgets = {row["id"]: row["seconds"] for row in tests}
    def weight(group):
        return sum(catalog[n]["seconds"] + .015 for n in group)

    warm()
    errors = 0
    with Session(warm_fixtures=False) as cold, Session() as hot:
        # The selector mounts the catalog at /data. Explicitly supply it when
        # checking multiple datasets in one container.
        hot.seconds = cold.seconds = budgets
        startup = hot.warm()
        assert startup > 0, "warm parents unavailable"
        print(json.dumps({"startup": startup}), flush=True)
        for digest, (raw, failing, names) in sorted(patches.items())[index::shards]:
            patch = raw.decode("utf-8", "surrogateescape")
            candidates = set(numbers(analyzer.analyze(patch, prioritize=True).mask))
            passing = sorted(catalog.keys() - failing)
            rng = random.Random(int.from_bytes(digest))
            sample = failing | set(rng.sample(passing, min(30, len(passing))))
            group = min((sample, candidates | failing), key=weight) if candidates else sample
            ids = [catalog[n]["id"] for n in sorted(group)]
            results = [s.run(patch, ids, per_test_budget=budgets) for s in (cold, hot)]
            a, b = results
            diff = a.failing ^ b.failing
            complete = all(len(r.ran) == len(ids) and "incomplete child run" not in r.notes for r in results)
            initial = sorted(diff)
            if diff == {GENERATOR_MEMORY} and complete:
                isolated = [s.run(patch, [GENERATOR_MEMORY], per_test_budget=budgets) for s in (cold, hot)]
                diff = isolated[0].failing ^ isolated[1].failing
                complete = all(r.ran == [GENERATOR_MEMORY] and "incomplete child run" not in r.notes
                               for r in isolated)
            errors += bool(diff or not complete)
            print(json.dumps(dict(commits=names, selected=len(ids), cold=a.cpu_s, warm=b.cpu_s,
                                  initial=initial, diff=sorted(diff), complete=complete,
                                  notes=[r.notes[:3] for r in results] if diff or not complete else [])), flush=True)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
