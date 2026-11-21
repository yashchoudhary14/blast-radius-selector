"""Offline collector of per-test and import-time reads on class-level dict literals.

Run in the pinned image with /data/tests.json and a writable /app/maps:
    python -m selector.dict_reads
"""

import argparse
import ast
import atexit
import builtins
import importlib
import importlib.abc
import importlib.machinery
import json
import marshal
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from importlib.machinery import SourceFileLoader
from pathlib import Path

from .collect import BASE, MAPS

_PURE_KEY_NODES = (ast.Constant, ast.Attribute, ast.Name, ast.Tuple, ast.UnaryOp, ast.BinOp, ast.Load)


def class_dict_sites(base=BASE):
    """Return {abs_file_path: {dict_loc: (did, [(ast_idx, ast_key_node), ...])}}."""
    by_file = {}
    seen_dids = set()
    ambiguous = set()
    raw = []

    for path in sorted((base / "sqlglot").rglob("*.py")):
        if path.name == "__main__.py":
            continue
        rel = path.relative_to(base).as_posix()
        tree = ast.parse(path.read_bytes(), str(path))

        def walk_cls(node, prefix=()):
            for stmt in node.body:
                if isinstance(stmt, ast.ClassDef):
                    cname = ".".join((*prefix, stmt.name))
                    for item in stmt.body:
                        if isinstance(item, (ast.Assign, ast.AnnAssign)) and item.value is not None:
                            targets = item.targets if isinstance(item, ast.Assign) else [item.target]
                            if (
                                isinstance(item.value, ast.Dict)
                                and len(targets) == 1
                                and isinstance(targets[0], ast.Name)
                            ):
                                did = (rel, cname, targets[0].id)
                                if did in seen_dids:
                                    ambiguous.add(did)
                                seen_dids.add(did)
                                raw.append((str(path), did, item.value))
                    walk_cls(stmt, (*prefix, stmt.name))

        walk_cls(tree)

    for fpath, did, dnode in raw:
        if did in ambiguous:
            continue
        key_dumps = []
        key_pairs = []
        valid = True
        for idx, k in enumerate(dnode.keys):
            if k is None:
                continue
            if not all(isinstance(sub, _PURE_KEY_NODES) for sub in ast.walk(k)):
                valid = False
                break
            dump = ast.dump(k)
            if dump in key_dumps:
                valid = False
                break
            key_dumps.append(dump)
            key_pairs.append((idx, k))
        if not valid:
            continue
        loc = (dnode.lineno, dnode.col_offset, dnode.end_lineno, dnode.end_col_offset)
        by_file.setdefault(fpath, {})[loc] = (did, key_pairs)

    return by_file


class _CollectorState:
    __slots__ = ("import_stack", "test_keys", "test_whole", "import_keys", "import_whole")

    def __init__(self):
        self.import_stack = []
        self.test_keys = defaultdict(set)
        self.test_whole = set()
        self.import_keys = defaultdict(lambda: defaultdict(set))
        self.import_whole = defaultdict(set)

    def clear_test(self):
        self.test_keys.clear()
        self.test_whole.clear()

    def clear_all(self):
        self.test_keys.clear()
        self.test_whole.clear()
        self.import_keys.clear()
        self.import_whole.clear()


_STATE = _CollectorState()


class TrackedDict(dict):
    __slots__ = ("_br_id", "_br_key_map")

    def __new__(cls, br_id=None, mapping=(), key_pairs=()):
        return super().__new__(cls)

    def __init__(self, br_id=None, mapping=(), key_pairs=()):
        if type(mapping) is not dict and isinstance(mapping, dict):
            mapping = dict.items(mapping)
        dict.__init__(self, mapping)
        self._br_id = br_id
        kmap = {}
        for idx, k in key_pairs:
            kmap[k] = idx
        self._br_key_map = kmap

    def _rec_key(self, k):
        did = self._br_id
        if did is None:
            return
        try:
            idx = self._br_key_map.get(k)
        except TypeError:
            return
        if idx is None:
            return
        stack = _STATE.import_stack
        if stack:
            _STATE.import_keys[did][idx].add(stack[-1])
        else:
            _STATE.test_keys[did].add(idx)

    def _rec_key_iter(self):
        did = self._br_id
        if did is None:
            return
        if not _STATE.import_stack:
            _STATE.test_whole.add(did)

    def _rec_whole(self):
        did = self._br_id
        if did is None:
            return
        stack = _STATE.import_stack
        if stack:
            _STATE.import_whole[did].add(stack[-1])
        else:
            _STATE.test_whole.add(did)

    def __getitem__(self, k):
        self._rec_key(k)
        return dict.__getitem__(self, k)

    def get(self, k, default=None):
        self._rec_key(k)
        return dict.get(self, k, default)

    def pop(self, *args):
        if args:
            self._rec_key(args[0])
        else:
            self._rec_whole()
        return dict.pop(self, *args)

    def setdefault(self, k, default=None):
        self._rec_key(k)
        return dict.setdefault(self, k, default)

    def __contains__(self, k):
        return dict.__contains__(self, k)

    def __bool__(self):
        return dict.__len__(self) != 0

    def __len__(self):
        self._rec_key_iter()
        return dict.__len__(self)

    def __iter__(self):
        self._rec_key_iter()
        return dict.__iter__(self)

    def keys(self):
        self._rec_key_iter()
        return dict.keys(self)

    def values(self):
        self._rec_whole()
        return dict.values(self)

    def items(self):
        self._rec_whole()
        return dict.items(self)

    def __reversed__(self):
        self._rec_whole()
        return dict.__reversed__(self)

    def __eq__(self, other):
        self._rec_whole()
        return dict.__eq__(self, other)

    def __ne__(self, other):
        self._rec_whole()
        return dict.__ne__(self, other)

    def __or__(self, other):
        self._rec_whole()
        return dict.__or__(self, other)

    def __ror__(self, other):
        self._rec_whole()
        return dict.__ror__(self, other)

    def __ior__(self, other):
        self._rec_whole()
        return dict.__ior__(self, other)

    def copy(self):
        self._rec_whole()
        return dict.copy(self)

    def update(self, *args, **kwargs):
        self._rec_whole()
        return dict.update(self, *args, **kwargs)

    def popitem(self):
        self._rec_whole()
        return dict.popitem(self)

    def clear(self):
        self._rec_whole()
        return dict.clear(self)

    def __setitem__(self, k, v):
        self._rec_whole()
        return dict.__setitem__(self, k, v)

    def __delitem__(self, k):
        self._rec_whole()
        return dict.__delitem__(self, k)

    def __repr__(self):
        self._rec_whole()
        return dict.__repr__(self)

    def __str__(self):
        self._rec_whole()
        return dict.__str__(self)

    def __format__(self, spec):
        self._rec_whole()
        return dict.__format__(self, spec)

    def __sizeof__(self):
        self._rec_whole()
        return dict.__sizeof__(self)

    def __reduce__(self):
        self._rec_whole()
        return (dict, (dict(dict.items(self)),))

    def __reduce_ex__(self, protocol):
        self._rec_whole()
        return (dict, (dict(dict.items(self)),))

    def __getstate__(self):
        self._rec_whole()
        return dict.__getstate__(self)


def _br_wrap_dict(did, mapping, key_pairs):
    return TrackedDict(did, mapping, key_pairs)


class _DictInstrument(importlib.abc.MetaPathFinder):
    def __init__(self, sites):
        self.sites = sites

    def find_spec(self, fullname, path, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec and spec.origin in self.sites:
            points = self.sites[spec.origin]

            def get_code(_name):
                tree = ast.parse(Path(spec.origin).read_bytes(), spec.origin)

                class Wrap(ast.NodeTransformer):
                    def visit_Dict(self, node):
                        node = self.generic_visit(node)
                        loc = (node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)
                        entry = points.get(loc)
                        if entry:
                            did, key_pairs = entry
                            pairs_ast = ast.Tuple(
                                [
                                    ast.Tuple([ast.Constant(idx), ast.parse(ast.unparse(k), mode="eval").body], ast.Load())
                                    for idx, k in key_pairs
                                ],
                                ast.Load(),
                            )
                            return ast.copy_location(
                                ast.Call(
                                    ast.Name("_br_wrap_dict", ast.Load()),
                                    [ast.Constant(did), node, pairs_ast],
                                    [],
                                ),
                                node,
                            )
                        return node

                return compile(ast.fix_missing_locations(Wrap().visit(tree)), spec.origin, "exec")

            spec.loader.get_code = get_code
            return spec
        return None


_INSTALLED = False


def install_instrumentation():
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    builtins._br_wrap_dict = _br_wrap_dict
    sites = class_dict_sites(BASE)
    sys.meta_path.insert(0, _DictInstrument(sites))
    orig_exec = SourceFileLoader.exec_module

    def tracked_exec(loader, module):
        path = Path(loader.path)
        if path.is_relative_to(BASE):
            rel = path.relative_to(BASE).as_posix()
            _STATE.import_stack.append(rel)
            try:
                return orig_exec(loader, module)
            finally:
                _STATE.import_stack.pop()
        return orig_exec(loader, module)

    SourceFileLoader.exec_module = tracked_exec


def _serialize_state():
    return {
        "test_keys": {did: sorted(idxs) for did, idxs in _STATE.test_keys.items()},
        "test_whole": list(_STATE.test_whole),
        "import_keys": {
            did: {idx: sorted(mods) for idx, mods in idxs.items()}
            for did, idxs in _STATE.import_keys.items()
        },
        "import_whole": {did: sorted(mods) for did, mods in _STATE.import_whole.items()},
    }


def install_child():
    """Called by temporary sitecustomize.py in spawned child processes."""
    log_dir = os.environ.get("BR_DICT_LOG_DIR")
    if not log_dir:
        return
    install_instrumentation()

    def _flush():
        target = Path(log_dir) / f"child_{os.getpid()}.bin"
        try:
            target.write_bytes(marshal.dumps(_serialize_state()))
        except OSError:
            pass

    atexit.register(_flush)


def measure_preload(output):
    install_instrumentation()
    os.chdir(BASE)
    for path in sorted((BASE / "sqlglot").rglob("*.py")):
        if path.name == "__main__.py":
            continue
        rel = path.relative_to(BASE).as_posix()
        mod = rel[:-3].replace("/", ".").removesuffix(".__init__")
        importlib.import_module(mod)
    output.write_bytes(marshal.dumps(_serialize_state()))


def measure_file(file, tests, output, site_dir):
    install_instrumentation()
    os.chdir(BASE)
    module = importlib.import_module(file[:-3].replace("/", "."))
    file_imports = {
        "import_keys": {
            did: {idx: sorted(mods) for idx, mods in idxs.items()}
            for did, idxs in _STATE.import_keys.items()
        },
        "import_whole": {did: sorted(mods) for did, mods in _STATE.import_whole.items()},
    }
    _STATE.clear_all()

    base_pypath = os.environ.get("PYTHONPATH", "")
    child_pypath = str(site_dir) + (os.pathsep + base_pypath if base_pypath else "")

    per_test = {}
    incomplete = []
    merged_import_keys = defaultdict(lambda: defaultdict(set))
    merged_import_whole = defaultdict(set)

    def merge_imp(imp_k, imp_w):
        for did, idxs in imp_k.items():
            for idx, mods in idxs.items():
                merged_import_keys[did][idx].update(mods)
        for did, mods in imp_w.items():
            merged_import_whole[did].update(mods)

    merge_imp(file_imports["import_keys"], file_imports["import_whole"])

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for test in tests:
            if test["file"] != file:
                continue
            num = test["number"]
            _, klass, method = test["id"].split("::")
            tdir = tmp_path / f"t_{num}"
            tdir.mkdir()
            pid = os.fork()
            if not pid:
                try:
                    os.environ["BR_DICT_LOG_DIR"] = str(tdir)
                    os.environ["PYTHONPATH"] = child_pypath
                    _STATE.clear_all()
                    forked = False
                    spawned = 0

                    def audit(event, args):
                        nonlocal forked, spawned
                        if event == "os.fork":
                            forked = True
                        elif event in ("os.posix_spawn", "subprocess.Popen"):
                            spawned += 1

                    sys.addaudithook(audit)
                    with open(os.devnull, "w") as sink:
                        os.dup2(sink.fileno(), 1)
                        result = unittest.TestResult()
                        case = getattr(module, klass)(method)
                        unittest.TestSuite([case]).run(result)
                    ok = (
                        result.wasSuccessful()
                        and result.testsRun == 1
                        and not result.skipped
                        and not forked
                    )
                    payload = {
                        "ok": ok,
                        "spawned": spawned,
                        "state": _serialize_state(),
                    }
                    (tdir / f"main_{os.getpid()}.bin").write_bytes(marshal.dumps(payload))
                    os._exit(0)
                except BaseException as exc:
                    print(f"{test['id']}: {exc!r}", file=sys.stderr, flush=True)
                    os._exit(1)

            _, status = os.waitpid(pid, 0)
            mains = list(tdir.glob("main_*.bin"))
            if status or len(mains) != 1:
                incomplete.append(num)
                shutil.rmtree(tdir, ignore_errors=True)
                continue
            main_data = marshal.loads(mains[0].read_bytes())
            children = list(tdir.glob("child_*.bin"))
            if not main_data["ok"] or len(children) < main_data["spawned"]:
                incomplete.append(num)
                shutil.rmtree(tdir, ignore_errors=True)
                continue

            t_keys = defaultdict(set)
            t_whole = set()
            st = main_data["state"]
            for did, idxs in st["test_keys"].items():
                t_keys[did].update(idxs)
            t_whole.update(st["test_whole"])
            merge_imp(st["import_keys"], st["import_whole"])

            for child_file in children:
                cst = marshal.loads(child_file.read_bytes())
                for did, idxs in cst["test_keys"].items():
                    t_keys[did].update(idxs)
                t_whole.update(cst["test_whole"])
                merge_imp(cst["import_keys"], cst["import_whole"])

            per_test[num] = (
                {did: sorted(idxs) for did, idxs in t_keys.items()},
                list(t_whole),
            )
            shutil.rmtree(tdir, ignore_errors=True)

    out_data = {
        "per_test": per_test,
        "incomplete": incomplete,
        "import_keys": {
            did: {idx: sorted(mods) for idx, mods in idxs.items()}
            for did, idxs in merged_import_keys.items()
        },
        "import_whole": {did: sorted(mods) for did, mods in merged_import_whole.items()},
    }
    output.write_bytes(marshal.dumps(out_data))


def build(tests, manifest, jobs):
    files = sorted({t["file"] for t in tests})
    seconds = defaultdict(float)
    for t in tests:
        seconds[t["file"]] += t["seconds"]
    files.sort(key=seconds.get, reverse=True)

    dicts = {}  # did -> {"keys": {idx: mask}, "whole": mask, "import_keys": {idx: set}, "import_whole": set}
    incomplete_mask = 0

    def ensure(did):
        entry = dicts.get(did)
        if entry is None:
            entry = {
                "keys": defaultdict(int),
                "whole": 0,
                "import_keys": defaultdict(set),
                "import_whole": set(),
            }
            dicts[did] = entry
        return entry

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        site_dir = tmp_path / "site"
        site_dir.mkdir()
        (site_dir / "sitecustomize.py").write_text(
            "from selector.dict_reads import install_child" + chr(10) + "install_child()" + chr(10)
        )

        preload_out = tmp_path / "preload.bin"
        subprocess.run(
            [sys.executable, "-m", "selector.dict_reads", "--preload", "--output", str(preload_out)],
            check=True,
        )
        preload_data = marshal.loads(preload_out.read_bytes())
        for did, idxs in preload_data["import_keys"].items():
            ent = ensure(did)
            for idx, mods in idxs.items():
                ent["import_keys"][idx].update(mods)
        for did, mods in preload_data["import_whole"].items():
            ensure(did)["import_whole"].update(mods)

        def run_file(file):
            out = tmp_path / (file.replace("/", "_") + ".bin")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "selector.dict_reads",
                    "--file",
                    file,
                    "--output",
                    str(out),
                    "--manifest",
                    str(manifest),
                    "--site-dir",
                    str(site_dir),
                ],
                check=True,
            )
            return marshal.loads(out.read_bytes())

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            for res in pool.map(run_file, files):
                for num in res["incomplete"]:
                    incomplete_mask |= 1 << num
                for num, (t_keys, t_whole) in res["per_test"].items():
                    bit = 1 << num
                    for did, idxs in t_keys.items():
                        kmap = ensure(did)["keys"]
                        for idx in idxs:
                            kmap[idx] = kmap.get(idx, 0) | bit
                    for did in t_whole:
                        ensure(did)["whole"] |= bit
                for did, idxs in res["import_keys"].items():
                    ent = ensure(did)
                    for idx, mods in idxs.items():
                        ent["import_keys"][idx].update(mods)
                for did, mods in res["import_whole"].items():
                    ensure(did)["import_whole"].update(mods)

    # Also register every valid dict site even if never touched at import or test time.
    for points in class_dict_sites(BASE).values():
        for did, _ in points.values():
            ensure(did)

    serialized = {
        "incomplete": incomplete_mask,
        "dicts": {
            did: {
                "keys": dict(ent["keys"]),
                "whole": ent["whole"],
                "import_keys": {idx: sorted(mods) for idx, mods in ent["import_keys"].items()},
                "import_whole": sorted(ent["import_whole"]),
            }
            for did, ent in dicts.items()
        },
    }
    (MAPS / "dict-reads.bin").write_bytes(marshal.dumps(serialized))
    print(
        f"Wrote dict-reads.bin ({len(dicts)} dicts, incomplete={incomplete_mask.bit_count()})",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("/data/tests.json"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--file")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--site-dir", type=Path)
    args = parser.parse_args()
    if args.preload:
        measure_preload(args.output)
    elif args.file:
        catalog = json.loads(args.manifest.read_bytes())["tests"]
        measure_file(args.file, catalog, args.output, args.site_dir)
    else:
        catalog = json.loads(args.manifest.read_bytes())["tests"]
        build(catalog, args.manifest, args.jobs)
