"""Analyze independent patches. CLI emits commit candidates and optionally checks recall."""

import argparse
import ast
import json
import marshal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from runner.overlay import changes
from .scopes import scope_changed, snapshot


def numbers(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


@dataclass
class Candidates:
    direct: int = 0
    broad: int = 0
    priority: int = 0
    broken: bool = False
    python_paths: list = field(default_factory=list)

    @property
    def mask(self):
        return self.direct | self.broad


class Analyzer:
    def __init__(self, manifest, base=Path("/sqlglot"), maps=Path("/app/maps")):
        self.base, self.maps = base, maps
        self.index = marshal.loads((maps / "index.bin").read_bytes())
        self.tests = json.loads(manifest.read_bytes())["tests"]
        self.ids = {t["number"]: t["id"] for t in self.tests}
        self.num = {i: n for n, i in self.ids.items()}
        self.file_tests, self.class_tests, self.methods = (defaultdict(int) for _ in range(3))
        metadata = self.index["dialect_mentions"]
        for t in self.tests:
            key, method = t["id"].rsplit("::", 1)
            bit = 1 << t["number"]
            self.file_tests[t["file"]] |= bit
            for parent in metadata["test_mro"].get(key, [key]):
                self.class_tests[parent] |= bit
                self.methods[parent + "." + method] |= bit
        self.importers = defaultdict(int, self.index["lazy_imports"])
        for file, paths in self.index["collect_imports"].items():
            for path in paths:
                self.importers[path] |= self.file_tests[file]
        for file, (reads, directories) in self.index["collection_reads"].items():
            for key, paths in (("reads", reads), ("dirs", directories)):
                masks = self.index[key]
                for path in paths:
                    masks[path] = masks.get(path, 0) | self.file_tests[file]
        mentions = {d: sum(1 << self.num[i] for i in ids if i in self.num)
                    for d, ids in metadata["mentions"].items()}
        self.dialects = defaultdict(int)
        for dialect, children in metadata["descendants"].items():
            for child in children:
                self.dialects[dialect] |= mentions.get(child, 0)

    @lru_cache(maxsize=32)
    def old(self, path):
        cache = self.maps / "ast" / (path + ".bin")
        if cache.is_file():
            return marshal.loads(cache.read_bytes())
        old = self.base / path
        source = old.read_bytes() if old.is_file() else b""
        return snapshot(ast.parse(source, path), source)

    @lru_cache(maxsize=16)
    def line_hits(self, path):
        cache = self.maps / "ast" / (path + ".lines")
        return marshal.loads(cache.read_bytes()) if cache.is_file() else {}

    def changed_line_hits(self, path, revised):
        hits = self.line_hits(path)
        if not hits or revised is None:
            return 0
        old = (self.base / path).read_bytes().splitlines()
        new = revised.splitlines()
        # Most patches have a single short hunk in a large parser or test file.
        prefix, shared = 0, min(len(old), len(new))
        while prefix < shared and old[prefix] == new[prefix]:
            prefix += 1
        old_end, new_end = len(old), len(new)
        while old_end > prefix and new_end > prefix and old[old_end - 1] == new[new_end - 1]:
            old_end -= 1
            new_end -= 1
        mask = 0
        edits = SequenceMatcher(None, old[prefix:old_end], new[prefix:new_end])
        for op, start, end, _, _ in edits.get_opcodes():
            if op != "equal":
                for index in range(prefix + start + 1, prefix + end + 1):
                    mask |= hits.get(index, 0)
        return mask

    def analyze(self, patch, *, prioritize=False):
        out = Candidates()
        for path, new in changes(self.base, patch).items():
            p = Path(path)
            readers = self.index["reads"].get(path, 0)
            out.direct |= readers
            out.priority |= readers
            if new is None or not (self.base / path).exists():
                directory = p.parent
                while not (self.base / directory).is_dir():
                    directory = directory.parent
                out.broad |= self.index["dirs"].get(directory.as_posix(), 0)
            if p.suffix == ".py":
                out.python_paths.append(path)
                self.python(path, new, out)
                if prioritize and not out.broken:
                    out.priority |= self.changed_line_hits(path, new)
        return out

    def python(self, path, new, out):
        coverage = self.index["all_hits"].get(path, 0)
        try:
            if new is None:
                raise SyntaxError("deleted module")
            # ast.parse alone accepts constructs such as return outside a function.
            compile(new, path, "exec", dont_inherit=True)
            tree = ast.parse(new, path)
        except (ValueError, SyntaxError):
            out.broad |= coverage | self.importers[path] | self.file_tests[path]
            out.broken = True
            return
        old_funcs, old_scopes = self.old(path)
        new_funcs, new_scopes = snapshot(tree, new, old_funcs)
        dirty = {q for q in old_scopes.keys() | new_scopes.keys()
                 if scope_changed(old_scopes.get(q), new_scopes.get(q))}
        hits = self.index["by_func"].get(path, {})
        runtime = self.index["runtime"].get(path, 0)
        test_file = path.startswith("tests/")
        changed = False
        for q in old_funcs.keys() | new_funcs.keys():
            old, revised = old_funcs.get(q), new_funcs.get(q)
            if old and revised and old[2] == revised[2]:
                continue
            changed = True
            owner, _, method = q.rpartition(".")
            if old:
                out.direct |= hits.get(q, 0)
            else:
                same = self.index["by_name"].get(method.split("#")[0], 0)
                out.broad |= (same & runtime) or same
            if old and revised and old[1] != revised[1]:
                dirty.add(owner)
            if old and not revised and not owner:
                # A removed binding can break a from-import in a test that
                # never invoked the function on the baseline.
                out.broad |= self.importers[path]
            if test_file:
                methods = self.methods[path + "::" + q]
                out.direct |= methods
                out.priority |= methods
                if method in {"setUp", "tearDown", "setUpClass", "tearDownClass",
                              "setUpModule", "tearDownModule"}:
                    dirty.add(owner)
        if changed or dirty:
            # Process-wide heap assertions can see allocations from imported
            # modules even when the test never calls the edited function.
            out.broad |= self.index["heap_observers"] & self.importers[path]
        if not dirty:
            return
        if test_file:
            for owner in dirty:
                affected = self.class_tests[path + "::" + owner] if owner else 0
                out.broad |= affected or (self.file_tests[path] | coverage | self.importers[path])
            return
        dialect = Path(path).stem
        out.broad |= runtime
        if path.startswith("sqlglot/") and dialect in self.dialects:
            out.broad |= self.dialects[dialect] | self.index["lazy_imports"].get(path, 0)
        else:
            # Class data can be read through inherited functions defined in
            # other modules, including while dependent modules are imported.
            out.broad |= coverage | self.importers[path]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    start = time.process_time()
    analyzer = Analyzer(args.dataset / "tests.json")
    print(f"map_load_cpu={time.process_time()-start:.4f}", file=sys.stderr)
    answers = json.loads((args.dataset / "answers.json").read_bytes())["commits"] if args.check else {}
    hits = failures = selected = 0
    for patch in sorted((args.dataset / "commits").glob("*.patch")):
        try:
            result = analyzer.analyze(patch.read_bytes().decode("utf-8", "surrogateescape"))
            candidates = list(numbers(result.mask))
        except Exception as e:
            print(f"fallback {patch.stem} {e!r}", file=sys.stderr)
            candidates = list(analyzer.ids)
        print(json.dumps({"commit": patch.stem, "candidates": candidates}))
        if args.check:
            failing = set(answers[patch.stem]["failing"])
            missing = failing.difference(candidates)
            if missing:
                print(f"miss {patch.stem}: {[(n, analyzer.ids[n]) for n in sorted(missing)]}", file=sys.stderr)
            failures += len(failing)
            selected += len(candidates)
            hits += len(failing) - len(missing)
    if args.check:
        print(f"recall={hits}/{failures} precision={hits}/{selected}", file=sys.stderr)
        return int(hits != failures)


if __name__ == "__main__":
    raise SystemExit(main())
