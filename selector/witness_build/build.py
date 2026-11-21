"""Build a compact per-test certificate of module execution and entry-argument invariants."""
import gzip
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def collect(test):
    file, cls, method = test["id"].split("::")
    node = ".".join([file[:-3].replace("/", "."), cls, method])
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, PYTHONPATH=f"{Path(__file__).parent}:/sqlglot", WITNESS_DIR=tmp)
        subprocess.run([sys.executable, "-m", "unittest", node], cwd="/sqlglot", env=env, check=True)
        modules, calls, inspected = set(), {}, set()
        lines, reads, comparisons = {}, set(), {}
        records = list(Path(tmp).iterdir())
        if not records:
            raise ValueError("collector did not start")
        for file in records:
            paths, obs, meta, covered, opened, outcomes = json.loads(file.read_bytes())
            for site, safe in outcomes.items():
                comparisons[site] = comparisons.get(site, True) and safe
            for path, visited in covered.items():
                lines.setdefault(path.removeprefix("/sqlglot/"), set()).update(visited)
            reads.update(p.removeprefix("/sqlglot/") for p in opened)
            modules.update(p.removeprefix("/sqlglot/") for p in paths)
            inspected.update(".".join(v) for v in meta if all(isinstance(s, str) for s in v))
            for key, args in obs.items():
                prior = calls.setdefault(key, args.copy())
                for p in list(prior):
                    prior[p] &= args.get(p, 0)
        return [sorted(modules), calls, sorted(inspected), {k: sorted(v) for k,v in lines.items()}, sorted(reads), sorted(k for k,v in comparisons.items() if v)]


if __name__ == "__main__":
    manifest = json.loads(Path("/data/tests.json").read_bytes())["tests"]
    dest = Path(sys.argv[1])
    for n in (1082, 1106, 1107, 1123, 1361):
        result = collect(manifest[n])
        raw = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
        (dest / f"calls-{n}.json.gz").write_bytes(gzip.compress(raw, mtime=0))
