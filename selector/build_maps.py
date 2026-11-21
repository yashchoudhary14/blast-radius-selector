"""Compile baseline coverage and structural signatures for the submission image."""

import ast
import gzip
import json
import marshal
from collections import defaultdict
from pathlib import PurePosixPath

from .collect import BASE, MAPS
from .scopes import snapshot


def read_gz(name):
    with gzip.open(MAPS / name) as src:
        return json.load(src)


def bits(numbers):
    return sum(1 << n for n in numbers)


def build():
    coverage = read_gz("lines.json.gz")
    bitsets = [bits(row) for row in coverage["sets"]]
    modlevel = read_gz("modlevel.json.gz")
    all_hits, runtime, by_func, by_name = {}, {}, {}, defaultdict(int)
    for path, lookup in coverage["files"].items():
        module = set(modlevel[path])
        all_hits[path] = 0
        nonmodule = {}
        for n, idx in lookup.items():
            line = int(n)
            all_hits[path] |= bitsets[idx]
            if line not in module:
                nonmodule[line] = bitsets[idx]
        runtime[path] = 0
        for mask in set(nonmodule.values()):
            runtime[path] |= mask
        source = (BASE / path).read_bytes()
        funcs, scopes = snapshot(ast.parse(source, path), source)
        cache = MAPS / "ast" / (path + ".bin")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(marshal.dumps((funcs, scopes)))
        # Keep line-level priorities per file so startup only reads the small
        # function index. The interning in the coverage map is preserved here.
        priority = {int(n): bitsets[idx] for n, idx in lookup.items()}
        cache.with_suffix(".lines").write_bytes(marshal.dumps(priority))
        by_func[path] = {}
        for name, (_, _, _, start, end) in funcs.items():
            mask = 0
            for line in range(start, end + 1):
                mask |= nonmodule.get(line, 0)
            if start == end and str(start) in lookup:
                mask |= bitsets[lookup[str(start)]]
            by_func[path][name] = mask
            if path.startswith("sqlglot/"):
                by_name[name.rsplit(".", 1)[-1].split("#")[0]] |= mask

    reads = {p: bits(ns) for p, ns in read_gz("reads.json.gz").items()}
    # Only remove opens which the independent audit observed in the bytecode
    # loader, with no corresponding source-as-data read in that test/fixture.
    for path, ns in json.loads((MAPS / "import_only_reads.json").read_bytes()).items():
        reads[path] &= ~bits(ns)
    dirs = defaultdict(int)
    for path, mask in reads.items():
        for parent in PurePosixPath(path).parents:
            dirs[str(parent)] |= mask
    for path, ns in json.loads((MAPS / "directory_reads.json").read_bytes()).items():
        dirs[path] |= bits(ns)

    lazy = json.loads((MAPS / "runtime_imports.json").read_bytes())
    index = dict(all_hits=all_hits, runtime=runtime, by_func=by_func, by_name=dict(by_name),
                 reads=reads, dirs=dict(dirs), heap_observers=bits(lazy["heap_observers"]),
                 lazy_imports={p: bits(ns) for p, ns in lazy["imports"].items()})
    for name in ("collect_imports", "collection_reads", "dialect_mentions", "durations"):
        index[name] = json.loads((MAPS / (name + ".json")).read_bytes())
    (MAPS / "index.bin").write_bytes(marshal.dumps(index))


if __name__ == "__main__":
    build()
