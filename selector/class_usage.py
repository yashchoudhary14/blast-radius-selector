"""Standalone collector for entry/maps/class_usage (compact bitsets)."""

import ast
import gzip
import importlib
import json
import marshal
import os
import sys
import time
import unittest
from collections import defaultdict
from pathlib import Path

from .collect import BASE, MAPS


def bits(nums):
    mask = 0
    for n in nums:
        mask |= 1 << n
    return mask


def build(manifest_path=Path("/data/tests.json")):
    start = time.monotonic()
    os.chdir(BASE)
    if str(BASE) not in sys.path:
        sys.path.insert(0, str(BASE))
    os.environ["SKIP_INTEGRATION"] = "0"
    sys.dont_write_bytecode = True

    catalog = json.loads(manifest_path.read_bytes())["tests"]

    # Import ONLY the base classes; do NOT pre-import dialects before running tests.
    from sqlglot.dialects.dialect import Dialect, _Dialect
    from sqlglot.expressions.core import Expr
    from sqlglot.generator import Generator
    from sqlglot.lineage import GraphHTML, Node
    from sqlglot.optimizer.annotate_types import TypeAnnotator
    from sqlglot.optimizer.scope import Scope
    from sqlglot.optimizer.simplify import Simplifier
    from sqlglot.parser import Parser
    from sqlglot.planner import Plan, Step
    from sqlglot.schema import AbstractMappingSchema, Schema
    from sqlglot.tokenizer_core import TokenizerCore
    from sqlglot.tokens import _TokenizerBase

    mro_cache = {}

    def get_mro_keys(cls):
        res = mro_cache.get(cls)
        if res is not None:
            return res
        keys = set()
        for base in getattr(cls, "__mro__", ()):
            mod = getattr(base, "__module__", "") or ""
            if mod.startswith("sqlglot"):
                mod_obj = sys.modules.get(mod)
                f = getattr(mod_obj, "__file__", None)
                if f:
                    try:
                        rel = Path(f).resolve().relative_to(BASE).as_posix()
                        keys.add(f"{rel}::{base.__qualname__}")
                    except ValueError:
                        pass
        mro_cache[cls] = frozenset(keys)
        return mro_cache[cls]

    active_classes = set()

    def record_cls(cls):
        if cls not in active_classes:
            active_classes.add(cls)

    bases_to_hook = [
        Expr, Dialect, _TokenizerBase, TokenizerCore, Parser, Generator,
        Schema, AbstractMappingSchema, Scope, TypeAnnotator, Simplifier,
        Plan, Step, Node, GraphHTML,
    ]

    for b in bases_to_hook:
        orig_init = b.__dict__.get("__init__")
        if orig_init is not None:
            def make_init(orig):
                def _init(self, *args, **kwargs):
                    record_cls(type(self))
                    return orig(self, *args, **kwargs)
                return _init
            setattr(b, "__init__", make_init(orig_init))
        else:
            orig_new = b.__dict__.get("__new__") or object.__new__
            def make_new(orig):
                def _new(cls, *args, **kwargs):
                    record_cls(cls)
                    if orig is object.__new__:
                        return orig(cls)
                    return orig(cls, *args, **kwargs)
                return _new
            setattr(b, "__new__", make_new(orig_new))

    orig_get = _Dialect.get.__func__
    @classmethod
    def hooked_get(cls, key, default=None):
        res = orig_get(cls, key, default)
        if res is not None:
            record_cls(res)
            for attr in ("tokenizer_class", "jsonpath_tokenizer_class", "parser_class", "generator_class"):
                sub = getattr(res, attr, None)
                if isinstance(sub, type):
                    record_cls(sub)
        return res
    _Dialect.get = hooked_get

    orig_getitem = _Dialect.__getitem__.__func__
    @classmethod
    def hooked_getitem(cls, key):
        res = orig_getitem(cls, key)
        if res is not None:
            record_cls(res)
            for attr in ("tokenizer_class", "jsonpath_tokenizer_class", "parser_class", "generator_class"):
                sub = getattr(res, attr, None)
                if isinstance(sub, type):
                    record_cls(sub)
        return res
    _Dialect.__getitem__ = hooked_getitem

    orig_dialect_init = Dialect.__init__
    def dialect_init(self, *args, **kwargs):
        cls = type(self)
        record_cls(cls)
        for attr in ("tokenizer_class", "jsonpath_tokenizer_class", "parser_class", "generator_class"):
            sub = getattr(cls, attr, None)
            if isinstance(sub, type):
                record_cls(sub)
        return orig_dialect_init(self, *args, **kwargs)
    Dialect.__init__ = dialect_init

    by_module = defaultdict(lambda: defaultdict(list))
    for item in catalog:
        mod_name = item["file"].removesuffix(".py").replace("/", ".")
        _, cls_name, method_name = item["id"].split("::")
        by_module[mod_name][cls_name].append((item["number"], method_name))

    usage_by_key = defaultdict(set)

    for mod_name, classes in by_module.items():
        active_classes.clear()
        mod = importlib.import_module(mod_name)
        mod_classes = set(active_classes)

        if hasattr(mod, "setUpModule"):
            active_classes.clear()
            mod.setUpModule()
            mod_classes |= active_classes

        for cls_name, methods in classes.items():
            test_cls = getattr(mod, cls_name)
            cls_classes = set(mod_classes)
            if hasattr(test_cls, "setUpClass"):
                active_classes.clear()
                test_cls.setUpClass()
                cls_classes |= active_classes
            try:
                for num, method_name in methods:
                    active_classes.clear()
                    case = test_cls(method_name)
                    result = unittest.TestResult()
                    case(result)
                    all_used = cls_classes | active_classes
                    for c in all_used:
                        for k in get_mro_keys(c):
                            usage_by_key[k].add(num)
            finally:
                if hasattr(test_cls, "tearDownClass"):
                    test_cls.tearDownClass()
        if hasattr(mod, "tearDownModule"):
            mod.tearDownModule()

    # Also union method coverage (including classmethods/staticmethods) across MRO
    import pkgutil
    import sqlglot
    for info in pkgutil.walk_packages(sqlglot.__path__, "sqlglot."):
        if info.name != "sqlglot.__main__":
            importlib.import_module(info.name)

    with gzip.open(MAPS / "lines.json.gz") as f:
        cov = json.load(f)
    with gzip.open(MAPS / "modlevel.json.gz") as f:
        modlevel = json.load(f)
    sets = cov["sets"]
    files_cov = cov["files"]

    method_cov = defaultdict(set)
    for p in sorted((BASE / "sqlglot").rglob("*.py")):
        rel = p.relative_to(BASE).as_posix()
        mod_lines = set(modlevel.get(rel, ()))
        lookup = files_cov.get(rel, {})
        line_tests = {int(ln): sets[idx] for ln, idx in lookup.items() if int(ln) not in mod_lines}
        tree = ast.parse(p.read_bytes())

        def visit_cls(node, prefix=()):
            for stmt in getattr(node, "body", ()):
                if isinstance(stmt, ast.ClassDef):
                    qname = ".".join((*prefix, stmt.name))
                    key = f"{rel}::{qname}"
                    for sub in ast.walk(stmt):
                        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            for ln in range(sub.lineno, (sub.end_lineno or sub.lineno) + 1):
                                method_cov[key].update(line_tests.get(ln, ()))
                    visit_cls(stmt, (*prefix, stmt.name))

        visit_cls(tree)

    # Propagate method coverage of subclasses to base classes via runtime MRO
    import inspect
    for modname, mod_obj in list(sys.modules.items()):
        if modname.startswith("sqlglot") and mod_obj is not None:
            for _, obj in inspect.getmembers(mod_obj, inspect.isclass):
                if obj.__module__.startswith("sqlglot"):
                    mro_keys = get_mro_keys(obj)
                    # Find obj's own key
                    f = getattr(sys.modules.get(obj.__module__), "__file__", None)
                    if f:
                        try:
                            rel = Path(f).resolve().relative_to(BASE).as_posix()
                            own_key = f"{rel}::{obj.__qualname__}"
                            own_tests = method_cov.get(own_key, set())
                            if own_tests:
                                for anc_key in mro_keys:
                                    usage_by_key[anc_key].update(own_tests)
                        except ValueError:
                            pass

    for k, ns in method_cov.items():
        usage_by_key[k].update(ns)

    bit_map = {k: bits(ns) for k, ns in sorted(usage_by_key.items()) if ns}
    (MAPS / "class_usage").write_bytes(marshal.dumps(bit_map))
    print(f"Wrote class_usage ({len(bit_map)} classes) in {time.monotonic() - start:.2f}s", file=sys.stderr)


if __name__ == "__main__":
    build()
