"""Group identical, stable observations by test bitmap, with one archive member per source."""
import gzip
import json
import marshal
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


def read(file):
    return marshal.loads(gzip.decompress(file.read_bytes()))


def pack(first, second, dest):
    sites = json.loads((first / "sites.json").read_bytes())
    assert sites == json.loads((second / "sites.json").read_bytes())
    keys = [(p, (line, col, kind, end)) for p, line, col, kind, end in sites]
    counts = Counter(keys)
    by_site = defaultdict(lambda: defaultdict(int))
    eligible = unstable = pairs = 0
    for file in sorted(first.iterdir()):
        if not file.name.isdecimal() or not (second / file.name).exists():
            continue
        bit = 1 << int(file.name)
        eligible |= bit
        a, a_bounds = read(file)
        b, b_bounds = read(second / file.name)
        for site in a.keys() | b.keys():
            av, bv = a.get(site), b.get(site)
            ar, br = a_bounds.get(site), b_bounds.get(site)
            # Type matters even when Python considers two scalar values equal.
            if av != bv or ar != br or ar and type(ar[0]) is not type(br[0]):
                signature = ((0,), None)
                unstable += 1
            elif 0 in av:
                signature = ((0,), None)
            else:
                signature = (tuple(sorted(av)), ar)
            # Equality of int/float range keys is harmless only for same-type
            # bound checks, so keep the type in the grouping key.
            tag = type(signature[1][0]).__name__ if signature[1] else ""
            by_site[site][(*signature, tag)] |= bit
            pairs += 1
    by_path = defaultdict(dict)
    for idx, (path, key) in enumerate(keys):
        if counts[(path, key)] == 1:
            by_path[path][key] = [(vals, bounds, mask)
                                  for (vals, bounds, _), mask in by_site[idx].items()]
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        def put(name, value):
            info = zipfile.ZipInfo(name)
            archive.writestr(info, marshal.dumps(value), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        put("complete", eligible)
        for path, rows in sorted(by_path.items()):
            put(path, rows)
    print(f"{len(sites)} sites, {eligible.bit_count()} completed twice, {pairs} observed pairs, "
          f"{unstable} seed disagreements, {dest.stat().st_size} bytes")


if __name__ == "__main__":
    pack(*map(Path, sys.argv[1:]))
