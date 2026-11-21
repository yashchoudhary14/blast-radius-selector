"""Analyze independent patches. CLI emits commit candidates and optionally checks recall."""

import argparse
import ast
import gzip
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

from .collect import BASE, MAPS
from .operands import narrow
from .query_cases import QueryCases, body_spans, frame_layout, pinned_layout
from .scopes import scope_changed, snapshot
from .source_equivalence import (compile_source, docstring_only, line_insertion, relocated_code,
                                 same_code, same_directives, signature_metadata)
from .state_refs import StateRefs, data_only, functions, same_runtime_declarations, signature

COMPOUND = (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith, ast.Try, ast.Match)
try:
    COMPOUND = (*COMPOUND, ast.TryStar)
except AttributeError:
    pass

INTERPRETER_SETTINGS = {"setrecursionlimit", "set_int_max_str_digits"}


def _get_func_node(tree, qname):
    curr = tree
    for part in qname.split("."):
        base_name, _, idx_str = part.partition("#")
        target_idx = int(idx_str) if idx_str else 1
        count = 0
        found = None
        if isinstance(curr, (ast.Module, ast.ClassDef)):
            children = curr.body
        else:
            children = (c for c in ast.walk(curr) if c is not curr)
        for stmt in children:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and stmt.name == base_name:
                count += 1
                if count == target_idx:
                    found = stmt
                    break
        if found is None:
            return None
        curr = found
    return curr


def _header_span(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
        end = node.body[0].lineno - 1 if node.body else (node.end_lineno or node.lineno)
        return start, max(start, end)
    if isinstance(node, COMPOUND):
        end = node.body[0].lineno - 1 if node.body else (node.end_lineno or node.lineno)
        return node.lineno, max(node.lineno, end)
    return node.lineno, node.end_lineno or node.lineno


def _collect_blocks_and_stmts(func_node):
    blocks, stmts = [], []

    def walk_block(header_sp, body):
        blocks.append((header_sp, body))
        for s in body:
            stmts.append(s)
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            h_sp = _header_span(s)
            if hasattr(s, "body") and s.body:
                walk_block(h_sp, s.body)
            if hasattr(s, "orelse") and s.orelse:
                walk_block(h_sp, s.orelse)
            if hasattr(s, "handlers") and s.handlers:
                for h in s.handlers:
                    stmts.append(h)
                    hh_end = h.body[0].lineno - 1 if h.body else (h.end_lineno or h.lineno)
                    walk_block((h.lineno, max(h.lineno, hh_end)), h.body)
            if hasattr(s, "finalbody") and s.finalbody:
                walk_block(h_sp, s.finalbody)
            if hasattr(s, "cases") and s.cases:
                for c in s.cases:
                    if c.body:
                        p_start = c.pattern.lineno
                        p_end = c.body[0].lineno - 1
                        walk_block((p_start, max(p_start, p_end)), c.body)

    walk_block(_header_span(func_node), func_node.body)
    return blocks, stmts


def _changed_stmt_spans(func_node, old_lines, new_lines, new_func_node):
    if func_node is None or new_func_node is None:
        return None
    f_start = min([func_node.lineno, *(d.lineno for d in func_node.decorator_list)])
    f_end = func_node.end_lineno or func_node.lineno
    nf_start = min([new_func_node.lineno, *(d.lineno for d in new_func_node.decorator_list)])
    nf_end = new_func_node.end_lineno or new_func_node.lineno

    old_slice = old_lines[f_start - 1 : f_end]
    new_slice = new_lines[nf_start - 1 : nf_end]

    blocks, stmts = _collect_blocks_and_stmts(func_node)
    def_h = _header_span(func_node)
    spans = []

    sm = SequenceMatcher(None, old_slice, new_slice)
    for tag, i1, i2, _, _ in sm.get_opcodes():
        if tag == "equal":
            continue
        if i1 < i2:
            for ln in range(f_start + i1, f_start + i2):
                if def_h[0] <= ln <= def_h[1]:
                    return [(f_start, f_end)]
                enclosing = [s for s in stmts if s.lineno <= ln <= (s.end_lineno or s.lineno)]
                if not enclosing:
                    return None
                smallest = min(enclosing, key=lambda s: (s.end_lineno or s.lineno) - s.lineno)
                if isinstance(smallest, (*COMPOUND, ast.ExceptHandler)):
                    if isinstance(smallest, ast.ExceptHandler):
                        h_end = smallest.body[0].lineno - 1 if smallest.body else (smallest.end_lineno or smallest.lineno)
                        spans.append((smallest.lineno, max(smallest.lineno, h_end)))
                    else:
                        spans.append(_header_span(smallest))
                else:
                    spans.append((smallest.lineno, smallest.end_lineno or smallest.lineno))
        else:
            ins_ln = f_start + i1
            best_block = None
            for h_sp, body in blocks:
                b_start = h_sp[0]
                b_end = (body[-1].end_lineno or body[-1].lineno) if body else h_sp[1]
                if b_start <= ins_ln <= b_end + 1:
                    if best_block is None or (b_end - b_start) <= (best_block[2] - best_block[1]):
                        best_block = (h_sp, b_start, b_end, body)
            if best_block is None:
                return None
            h_sp, _, _, body = best_block
            if h_sp != def_h:
                spans.append(h_sp)
            before_stmt = after_stmt = None
            first_doc = (body[0] if body and isinstance(body[0], ast.Expr)
                         and isinstance(body[0].value, ast.Constant)
                         and isinstance(body[0].value.value, str) else None)
            for s in body:
                if s is first_doc:
                    continue
                s_end = s.end_lineno or s.lineno
                if s_end < ins_ln:
                    before_stmt = s
                elif s.lineno >= ins_ln and after_stmt is None:
                    after_stmt = s
            if before_stmt is not None:
                spans.append((before_stmt.lineno, before_stmt.end_lineno or before_stmt.lineno))
            if after_stmt is not None:
                spans.append((after_stmt.lineno, after_stmt.end_lineno or after_stmt.lineno))
    return spans or None


@dataclass
class Candidates:
    direct: int = 0
    broad: int = 0
    priority: int = 0
    broken: bool = False
    structural: bool = False
    python_paths: list[str] = field(default_factory=list)
    dialect_subset_safe: bool = True
    query_spans: dict | None = field(default_factory=dict)
    query_subsets: dict = field(default_factory=dict)

    @property
    def mask(self):
        return self.direct | self.broad

    def selected(self):
        return set(numbers(self.mask)) - {n for n, indices in self.query_subsets.items() if not indices}


def numbers(mask):
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def collection_actions(source):
    """Find statements that can write shared state while a test module imports."""
    if source is None:
        return ()
    try:
        body = ast.parse(source).body
    except (SyntaxError, ValueError):
        return ()

    def writes(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return False
        if isinstance(node, ast.Expr):
            return any(isinstance(expr, ast.Call) for expr in ast.walk(node.value))
        if isinstance(node, (ast.Attribute, ast.Subscript)):
            return isinstance(node.ctx, (ast.Store, ast.Del))
        if isinstance(node, ast.Name):
            return node.id == "pytest_plugins" and isinstance(node.ctx, (ast.Store, ast.Del))
        return any(writes(child) for child in ast.iter_child_nodes(node))

    return tuple(ast.dump(stmt) for stmt in body if writes(stmt))


def changes_frame_bindings(before, after):
    if before is None or after is None:
        return True
    if (before.co_flags != after.co_flags or before.co_freevars != after.co_freevars
            or before.co_cellvars != after.co_cellvars):
        return True
    old_locals, new_locals = set(before.co_varnames), set(after.co_varnames)
    return (not (new_locals - old_locals).isdisjoint(before.co_names)
            or not (old_locals - new_locals).isdisjoint(after.co_names))


def interpreter_settings(source):
    """Find writes to interpreter-wide settings that survive a test's execution."""
    if not source or not any(name.encode() in source for name in INTERPRETER_SETTINGS):
        return ()
    try:
        nodes = tuple(ast.walk(ast.parse(source)))
    except (SyntaxError, ValueError):
        return ()
    modules, setters = set(), set()
    for node in nodes:
        if isinstance(node, ast.Import):
            modules.update(a.asname or a.name for a in node.names if a.name == "sys")
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == "sys":
            setters.update(a.asname or a.name for a in node.names if a.name in INTERPRETER_SETTINGS)
    return tuple(
        ast.dump(node)
        for node in nodes
        if isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Name) and node.func.id in setters
            or isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id in modules
            and node.func.attr in INTERPRETER_SETTINGS
        )
    )


class Analyzer:
    def __init__(self, manifest, base=Path("/sqlglot"), maps=Path("/app/maps")):
        self.base, self.maps = base, maps
        self.state_refs = StateRefs(self)
        self.index = marshal.loads((maps / "index.bin").read_bytes())
        dict_reads_path = maps / "dict-reads.bin"
        self.dict_reads = marshal.loads(dict_reads_path.read_bytes()) if dict_reads_path.is_file() else None
        self.tests = json.loads(manifest.read_bytes())["tests"]
        self.ids = {t["number"]: t["id"] for t in self.tests}
        self.num = {i: n for n, i in self.ids.items()}
        self.queries = QueryCases(self)
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
        imports_bin = maps / "import_lines.bin"
        if imports_bin.is_file():
            imports = marshal.loads(imports_bin.read_bytes())
        else:
            with gzip.open(maps / "import_lines.json.gz") as source:
                imports = json.load(source)
        masks = []
        for modules in imports["sets"]:
            mask = 0
            for module in modules:
                mask |= self.importers[module]
            masks.append(mask)
        self.import_hits = {path: {int(ln): masks[idx] for ln, idx in rows.items()}
                            for path, rows in imports["files"].items()}
        self.import_modules = {path: {int(ln): imports["sets"][idx] for ln, idx in rows.items()}
                               for path, rows in imports["files"].items()}
        self.indirect_hits = {path: dict(rows) for path, rows in self.import_hits.items()}
        self.cache_hits = {}
        # Lines executed only while their own module is imported (class/module bodies defining it).
        self.import_self_only = {path: {int(ln) for ln, idx in rows.items() if imports["sets"][idx] == [path]}
                                 for path, rows in imports["files"].items()}
        for cache in imports["caches"]:
            users = 0
            for path, ln in cache["reads"]:
                users |= self.line_hits(path).get(ln, 0) | self.indirect_hits.get(path, {}).get(ln, 0)
            for path, lines in cache["files"].items():
                rows = self.indirect_hits.setdefault(path, {})
                c_rows = self.cache_hits.setdefault(path, {})
                for ln in lines:
                    rows[ln] = rows.get(ln, 0) | users
                    c_rows[ln] = c_rows.get(ln, 0) | users
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

    @lru_cache(maxsize=None)
    def old(self, path):
        cache = self.maps / "ast" / (path + ".bin")
        if cache.is_file():
            return marshal.loads(cache.read_bytes())
        source = (self.base / path).read_bytes()
        return snapshot(ast.parse(source, path), source)

    @lru_cache(maxsize=64)
    def old_tree_and_lines(self, path):
        source = (self.base / path).read_bytes()
        return ast.parse(source, path), source.splitlines()

    def expr_owners(self, path, owners):
        """True if every owner is an Expression subclass (by base-class names across sqlglot/expressions)."""
        if getattr(self, "_expr_classes", None) is None:
            bases = {}
            for f in sorted((self.base / "sqlglot/expressions").glob("*.py")):
                for node in ast.walk(ast.parse(f.read_bytes(), str(f))):
                    if isinstance(node, ast.ClassDef):
                        bases.setdefault(node.name, set()).update(
                            b.id if isinstance(b, ast.Name) else b.attr if isinstance(b, ast.Attribute) else ""
                            for b in node.bases)
            expr = {"Expr", "Expression"}
            grew = True
            while grew:
                grew = False
                for name, names in bases.items():
                    if name not in expr and names & expr:
                        expr.add(name)
                        grew = True
            self._expr_classes = expr
        return bool(owners) and all(o.split(".")[-1] in self._expr_classes for o in owners)

    @lru_cache(maxsize=1)
    def expression_keys(self):
        masks = defaultdict(int)
        for key, mask in self.index.get("class_usage", {}).items():
            path, _, owner = key.partition("::")
            if path.startswith("sqlglot/expressions/") and self.expr_owners(path, {owner}):
                masks[owner.rsplit(".", 1)[-1].lower()] |= mask
        return masks

    @lru_cache(maxsize=32)
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

    @property
    def all_tests(self):
        mask = 0
        for t in self.tests:
            mask |= 1 << t["number"]
        return mask

    def _pytest_config(self, p, new):
        """Config files pytest discovers for the test tree."""
        dedicated = {"pytest.ini", ".pytest.ini", "pytest.toml", ".pytest.toml"}
        if p.name not in dedicated | {"tox.ini", "setup.cfg", "pyproject.toml"}:
            return False
        parent = p.parent.as_posix()
        if parent not in (".", "") and parent != "tests" and not parent.startswith("tests/"):
            return False
        if p.name in dedicated:
            return True
        old = (self.base / p).read_bytes() if (self.base / p).is_file() else b""
        return b"pytest" in old.lower() or b"pytest" in (new or b"").lower()

    @lru_cache(maxsize=32)
    def code_signatures(self, path):
        cache = self.maps / "ast" / (path + ".code")
        return marshal.loads(cache.read_bytes()) if cache.is_file() else {}

    @lru_cache(maxsize=32)
    def old_compiled(self, path):
        return compile_source((self.base / path).read_bytes(), path)

    @lru_cache(maxsize=32)
    def old_code(self, path):
        return functions(self.old_compiled(path)[0])

    @lru_cache(maxsize=1)
    def doc_observers(self):
        mask = 0
        # The pinned "*module" sites are lazy-export tables (which never
        # forward __doc__) and serde's immediately invoked attribute (where
        # both doc values are non-callable strings).
        for name in ("getdoc", "getsource", "getsourcelines", "DocTestSuite", "doctest",
                     "pydoc", "help", "*", "*sqlvalue", "co_positions", "__code__"):
            mask |= self.state_refs.mask(name)
        for path, lines in self.state_refs.refs.get("__doc__", {}).items():
            # This pinned hook reads only the doc of an Expr subclass under
            # construction. Plain top-level classes and modules can't be cls.
            hook = None
            if path == "sqlglot/expressions/core.py":
                hook = _get_func_node(self.old_tree_and_lines(path)[0], "Expr.__init_subclass__")
            runtime = self.line_hits(path)
            imports = self.indirect_hits.get(path, {})
            for line in lines:
                if hook is None or not hook.lineno <= line <= hook.end_lineno:
                    mask |= runtime.get(line, 0) | imports.get(line, 0)
        return mask

    @lru_cache(maxsize=1)
    def signature_observers(self):
        mask = 0
        for name in ("__annotations__", "__defaults__", "__kwdefaults__", "get_type_hints",
                     "get_annotations", "signature", "getargspec", "getfullargspec", "getsource",
                     "getsourcelines", "co_positions", "__code__", "pydoc", "help", "*", "*sqlvalue",
                     "exec", "eval", "locals", "f_locals", "currentframe", "_getframe", "getouterframes"):
            mask |= self.state_refs.mask(name)
        return mask

    @lru_cache(maxsize=1)
    def location_observers(self):
        mask = 0
        for name in ("getsource", "getsourcelines", "linecache", "__code__", "co_positions", "co_lines",
                     "co_firstlineno", "__firstlineno__", "co_names", "co_lnotab", "co_linetable",
                     "lineno", "f_lineno", "tb_lineno", "records", "traceback", "__traceback__", "exc_info",
                     "currentframe", "_getframe", "getouterframes", "settrace", "gettrace", "setprofile",
                     "catch_warnings", "filterwarnings", "assertWarns", "assertWarnsRegex", "warns", "*"):
            mask |= self.state_refs.mask(name)
        return mask

    def _source_equivalence(self, path, new, new_tree, compiled, headers, scope_edits):
        old = (self.base / path).read_bytes()
        insertion = line_insertion(old, new)
        if not (scope_edits or headers or insertion):
            return None
        before, diagnostics = self.old_compiled(path)
        if diagnostics:
            return None
        if insertion and relocated_code(before, compiled, insertion) and same_directives(old, new, insertion):
            return self.location_observers()
        if not (scope_edits or headers):
            return None
        if same_code(before, compiled) and same_directives(old, new):
            return 0  # Independent source readers were already selected.
        old_tree, _ = self.old_tree_and_lines(path)
        if docstring_only(old, new, old_tree, new_tree, before, compiled):
            return self.doc_observers()
        if len(headers) == 1:
            q = headers[0]
            if signature_metadata(old, new, old_tree, new_tree, q, before, compiled):
                body = functions(compiled).get(q)
                original = self.old_code(path).get(q)
                if body and original and same_code(original, body) and same_directives(old, new):
                    return self.signature_observers()
        return None

    def analyze(self, patch, *, prioritize=False, session=None, deadline=None):
        out = Candidates()
        edits = changes(self.base, patch)
        for directory in edits.removed_dirs:
            out.broad |= self.index["dirs"].get(Path(directory).parent.as_posix(), 0)
        from runner import Session
        @lru_cache(maxsize=1)
        def state():
            end = min(time.monotonic() + 5, deadline or float("inf"))
            if session:
                result = session().import_state(edits, end)
            else:
                with Session(base=self.base, warm_fixtures=False) as child:
                    result = child.import_state(edits, end)
            status, rows, cpu = result
            print(f"state {','.join(edits)} status={status} bindings={len(rows)} cpu={cpu:.4f}",
                  file=sys.stderr)
            return status, rows

        probe = state if all(p.startswith("sqlglot/") and p.endswith(".py")
                             and p != "sqlglot/__main__.py" and data is not None
                             and (self.base / p).is_file() for p, data in edits.items()) else None
        for path, new in edits.items():
            p = Path(path)
            readers = self.index["reads"].get(path, 0)
            if new is None or not (path.startswith("sqlglot/") and p.suffix == ".py"
                                   and (self.base / path).is_file()):
                out.query_spans = None
            out.direct |= readers
            out.priority |= readers
            created = new is not None and not (self.base / path).is_file()
            if new is None or created:
                out.structural = True
                out.dialect_subset_safe = False
                directory = p.parent
                while not (self.base / directory).is_dir():
                    out.broad |= self.index["missing_imports"].get(directory.as_posix() + ".py", 0)
                    directory = directory.parent
                out.broad |= self.index["dirs"].get(directory.as_posix(), 0)
            if self._pytest_config(p, new):
                # pytest reads these at startup (markers, addopts, filterwarnings=error, plugins): any test
                # can change outcome, and the unittest probes never load them.
                out.broad |= self.all_tests
                out.broken = True
                continue
            if p.suffix == ".py":
                out.python_paths.append(path)
                if not path.startswith("sqlglot/"):
                    out.dialect_subset_safe = False
                if path.startswith(("sqlglot/", "tests/")):
                    original = None if created else (self.base / path).read_bytes()
                    if interpreter_settings(original) != interpreter_settings(new):
                        # The full suite can observe writes from a module or test
                        # that an isolated execution probe never loads or runs.
                        out.broad |= self.all_tests
                        out.broken = True
                        continue
                if p.name == "conftest.py" and (p.parent == Path(".") or p.parts[0] == "tests"):
                    # Even a nested conftest can change the entire item list.
                    out.broad |= self.all_tests
                    out.broken = True
                    continue
                if path.startswith("tests/") and (p.match("test_*.py") or p.match("*_test.py")
                                                  or self.importers[path]):
                    if collection_actions(original) != collection_actions(new):
                        # Shared changes during full-suite collection are invisible to
                        # probes that import only the test modules they execute.
                        out.broad |= self.all_tests
                        out.broken = True
                        continue
                if created:
                    lookup = p.parent.as_posix() + ".py" if p.name == "__init__.py" else path
                    out.broad |= self.index["missing_imports"].get(lookup, 0)
                    if p.name == "__init__.py":
                        # A new package takes precedence over a sibling module of the same name.
                        out.broad |= self.importers[lookup] | self.index["all_hits"].get(lookup, 0)
                        # Imports through an existing namespace now execute its initializer.
                        prefix = p.parent.as_posix() + "/"
                        for module, mask in self.importers.items():
                            if module.startswith(prefix):
                                out.broad |= mask
                    continue
                if not path.startswith(("tests/", "sqlglot/")):
                    # Packaging scripts and examples may be read or executed by path.
                    out.broad |= self.importers[path] | self.index["all_hits"].get(path, 0)
                    continue
                self.python(path, new, out, prioritize=prioritize, probe=probe, single_edit=(len(edits) == 1),
                            operand_safe=len(edits) == 1)
        out.query_subsets = self.queries.subsets(out, edits)
        return out

    @lru_cache(maxsize=1)
    def _test_star_imports(self):
        stars = set()
        for file in sorted((self.base / "tests").rglob("*.py")):
            rel = file.relative_to(self.base).as_posix()
            tree, _ = self.old_tree_and_lines(rel)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
                    stars.add(_resolve_import_from(rel, node))
        return stars

    def _unused_added_imports(self, path, old_tree, new_tree, rows):
        old_dumps = [ast.dump(s) for s in old_tree.body]
        new_dumps = [ast.dump(s) for s in new_tree.body]
        sm = SequenceMatcher(None, old_dumps, new_dumps, autojunk=False)
        added_aliases = []  # (mod_name, bound_name)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            if tag == "insert":
                for stmt in new_tree.body[j1:j2]:
                    if isinstance(stmt, ast.Import):
                        for a in stmt.names:
                            added_aliases.append((a.name, a.asname or a.name.partition(".")[0]))
                    elif isinstance(stmt, ast.ImportFrom):
                        if any(a.name == "*" for a in stmt.names) or stmt.module == "__future__":
                            return False
                        mod = _resolve_import_from(path, stmt)
                        for a in stmt.names:
                            added_aliases.append((mod, a.asname or a.name))
                    else:
                        return False
            elif tag == "replace" and (i2 - i1) == 1 and (j2 - j1) == 1:
                o_stmt, n_stmt = old_tree.body[i1], new_tree.body[j1]
                if type(o_stmt) is not type(n_stmt) or not isinstance(o_stmt, (ast.Import, ast.ImportFrom)):
                    return False
                if isinstance(o_stmt, ast.ImportFrom):
                    if o_stmt.module != n_stmt.module or o_stmt.level != n_stmt.level or o_stmt.module == "__future__":
                        return False
                o_al = [ast.dump(a) for a in o_stmt.names]
                n_al = [ast.dump(a) for a in n_stmt.names]
                if any(a.name == "*" for a in n_stmt.names):
                    return False
                al_sm = SequenceMatcher(None, o_al, n_al, autojunk=False)
                for a_tag, _, _, aj1, aj2 in al_sm.get_opcodes():
                    if a_tag == "equal":
                        continue
                    if a_tag != "insert":
                        return False
                    mod = None if isinstance(n_stmt, ast.Import) else _resolve_import_from(path, n_stmt)
                    for a in n_stmt.names[aj1:aj2]:
                        added_aliases.append((a.name if mod is None else mod, a.asname or a.name.partition(".")[0] if mod is None else a.asname or a.name))
            else:
                return False
        if not added_aliases:
            return False
        old_imported = set()
        for stmt in old_tree.body:
            if isinstance(stmt, ast.Import):
                old_imported.update(a.name for a in stmt.names)
            elif isinstance(stmt, ast.ImportFrom):
                old_imported.add(_resolve_import_from(path, stmt))
        for mod, _ in added_aliases:
            if mod == "__future__" or not (mod.partition(".")[0] in sys.stdlib_module_names or mod in old_imported):
                return False
        added_names = {name for _, name in added_aliases}
        for node in ast.walk(old_tree):
            if isinstance(node, ast.Name):
                if node.id in added_names or node.id == "__all__":
                    return False
            elif isinstance(node, ast.alias):
                bound = node.asname or node.name.partition(".")[0]
                if bound in added_names or node.name in added_names:
                    return False
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name in added_names or node.name == "__getattr__":
                    return False
            elif isinstance(node, ast.arg) and node.arg in added_names:
                return False
            elif isinstance(node, (ast.Global, ast.Nonlocal)) and added_names.intersection(node.names):
                return False
            elif isinstance(node, ast.Call):
                fn = node.func
                callee = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
                if callee in ("globals", "locals", "vars", "dir"):
                    return False
        mod_name = path[:-3].replace("/", ".").removesuffix(".__init__")
        if not all(r[0] == mod_name and r[1] in added_names for r in rows):
            return False
        if mod_name in self._test_star_imports():
            return False
        for attr in added_names:
            if attr in self.state_refs.fields:
                return False
            for ref_path in self.state_refs.refs.get(attr, ()):
                if ref_path == path:
                    continue
                ref_tree, _ = self.old_tree_and_lines(ref_path)
                for node in ast.walk(ref_tree):
                    if isinstance(node, ast.ImportFrom) and any(a.name == attr for a in node.names):
                        target = _resolve_import_from(ref_path, node)
                        if target == mod_name or f"{target}.{attr}" == mod_name:
                            return False
        if path.startswith(("sqlglot/dialects/", "sqlglot/optimizer/")):
            return False
        return True

    def _dict_value_candidates(self, path, old_tree, new_bytes, rows):
        if not self.dict_reads:
            return None
        old_dicts, old_amb = _class_dict_nodes(old_tree)
        new_tree = ast.parse(new_bytes, path)
        new_dicts, new_amb = _class_dict_nodes(new_tree)
        if old_amb or new_amb or old_dicts.keys() != new_dicts.keys():
            return None
        tracked = self.dict_reads["dicts"]
        changed_dicts = {}
        for (cname, attr), od in old_dicts.items():
            nd = new_dicts[(cname, attr)]
            if len(od.keys) != len(nd.keys):
                return None
            did = (path, cname, attr)
            changed_idxs = []
            for idx, (ok, nk) in enumerate(zip(od.keys, nd.keys)):
                if ok is None or nk is None:
                    if ok is not None or nk is not None or ast.dump(od.values[idx]) != ast.dump(nd.values[idx]):
                        return None
                    continue
                if ast.dump(ok) != ast.dump(nk):
                    return None
                if ast.dump(od.values[idx]) != ast.dump(nd.values[idx]):
                    if did not in tracked:
                        return None
                    changed_idxs.append(idx)
            if changed_idxs:
                for idx in changed_idxs:
                    nd.values[idx] = od.values[idx]
                changed_dicts[did] = changed_idxs
        if not changed_dicts or ast.dump(old_tree) != ast.dump(new_tree):
            return None
        mod_name = path[:-3].replace("/", ".").removesuffix(".__init__")
        expected_owners = {(f"{mod_name}.{cname}", attr) for (_, cname, attr) in changed_dicts}
        has_import_value_reads = any(
            tracked[did]["import_whole"] or any(idx in tracked[did]["import_keys"] for idx in idxs)
            for did, idxs in changed_dicts.items()
        )
        for row in rows:
            if (row[0], row[1]) not in expected_owners and not has_import_value_reads:
                return None
        mask = self.dict_reads["incomplete"]
        for did, idxs in changed_dicts.items():
            entry = tracked[did]
            mask |= entry["whole"]
            kmap = entry["keys"]
            imp_k = entry["import_keys"]
            imp_mods = set(entry["import_whole"])
            for idx in idxs:
                mask |= kmap.get(idx, 0)
                if idx in imp_k:
                    imp_mods.update(imp_k[idx])
            for mod in imp_mods:
                mask |= self.importers.get(mod, 0) | self.file_tests.get(mod, 0)
        return mask

    def python(self, path, new, out, *, prioritize=False, probe=None, single_edit=False, operand_safe=True):
        coverage = self.index["all_hits"].get(path, 0)
        try:
            if new is None:
                raise SyntaxError("deleted module")
            # ast.parse alone accepts constructs such as return outside a function.
            compiled, diagnostics = compile_source(new, path)
            tree = ast.parse(new, path)
            new_tree_for_stmts = ast.parse(new, path)
        except (ValueError, SyntaxError):
            out.broad |= coverage | self.importers[path] | self.file_tests[path]
            out.broken = True
            return
        if diagnostics and diagnostics != self.old_compiled(path)[1]:
            # The grader can promote a compile-time warning to an import error.
            out.broad |= coverage | self.importers[path] | self.file_tests[path]
            out.broken = True
            return
        old_funcs, old_scopes = self.old(path)
        new_funcs, new_scopes = snapshot(tree, new, old_funcs)
        if single_edit and not diagnostics and path.startswith("sqlglot/"):
            headers = [q for q, fn in new_funcs.items() if old_funcs.get(q, (None, None))[1] != fn[1]]
            scope_edits = old_scopes != new_scopes
            if scope_edits or headers or any(old_funcs[q][2] != new_funcs[q][2] for q in old_funcs.keys() & new_funcs.keys()):
                refined = self._source_equivalence(path, new, new_tree_for_stmts, compiled, headers, scope_edits)
                if refined is not None:
                    out.direct |= refined
                    out.query_spans = None
                    out.dialect_subset_safe = False
                    return
        # Module statements, top-level definition order, and class/function
        # headers execute at import. Ignore changes confined to their bodies.
        if out.dialect_subset_safe:
            out.dialect_subset_safe = (
                old_scopes.get("") == new_scopes.get("")
                and all(old_scopes.get(q, (None,))[0] == scope[0]
                        for q, scope in new_scopes.items() if q and "." not in q)
                and all(old_funcs.get(q, (None, None))[1] == fn[1]
                        for q, fn in new_funcs.items() if "." not in q)
            )
        dirty = {q for q in old_scopes.keys() | new_scopes.keys()
                 if scope_changed(old_scopes.get(q), new_scopes.get(q))}
        scope_edit = bool(dirty)
        if old_scopes != new_scopes or old_funcs.keys() != new_funcs.keys():
            out.query_spans = None
        if dirty:
            # Class/module-level edits that sit entirely inside lambda bodies (e.g. entries of
            # Parser.FUNCTIONS / Generator.TRANSFORMS dicts) are code, not data: a lambda body runs
            # at call time, so per-test line coverage of those lines names every caller.
            base_src = (self.base / path).read_bytes() if (self.base / path).is_file() else b""
            old_lambdas, new_lambdas = _MaskLambdas(), _MaskLambdas()
            _, old_masked = snapshot(old_lambdas.visit(ast.parse(base_src, path)), base_src)
            _, new_masked = snapshot(new_lambdas.visit(ast.parse(new, path)), new)
            lambda_only = {q for q in dirty if q in old_masked and q in new_masked
                           and not scope_changed(old_masked[q], new_masked[q])}
            if lambda_only:
                dirty -= lambda_only
                l_hits = self.line_hits(path)
                touched = _changed_old_lines(base_src, new)
                for ln in touched:
                    out.direct |= l_hits.get(ln, 0)
                if old_lambdas.frames != new_lambdas.frames:
                    for start, end in old_lambdas.bodies:
                        if any(start <= ln <= end for ln in touched):
                            # A binding on one branch can invalidate a load
                            # in another branch of the same callback.
                            for ln in range(start, end + 1):
                                out.direct |= l_hits.get(ln, 0)
        hits = self.index["by_func"].get(path, {})
        runtime = self.index["runtime"].get(path, 0)
        test_file = path.startswith("tests/")
        # Another edit to a special method can alter the counterfactual operator
        # even if the old operator never called it.
        operand_safe = (operand_safe and not test_file and not scope_edit and old_funcs.keys() == new_funcs.keys()
                        and all(old_funcs[q][1] == new_funcs[q][1] and
                                (old_funcs[q][2] == new_funcs[q][2] or
                                 not any(p.startswith("__") and p.split("#")[0].endswith("__")
                                         for p in q.split("."))) for q in old_funcs))
        changed = False
        code = None
        equivalent = set()
        all_body_only = True
        header_skip = set()
        changed_symbols = set()
        mod_name = path[:-3].replace("/", ".").removesuffix(".__init__")

        @lru_cache(maxsize=1)
        def same_layout():
            return pinned_layout(self.base, path) == frame_layout(compiled)

        for q in old_funcs.keys() | new_funcs.keys():
            old, revised = old_funcs.get(q), new_funcs.get(q)
            if old and revised and old[2] == revised[2]:
                continue
            if old and revised and old[1] == revised[1]:
                if code is None:
                    code = functions(compiled)
                if code.get(q) and signature(code[q]) == self.code_signatures(path).get(q):
                    def docs(tree):
                        fn = _get_func_node(tree, q)
                        return [ast.get_docstring(n, clean=False) for n in ast.walk(fn)
                                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))] if fn else []
                    # Actual docstring edits can affect introspection/doctests.
                    # Preserve their normal source/statement attribution.
                    if docs(self.old_tree_and_lines(path)[0]) == docs(new_tree_for_stmts):
                        equivalent.update(range(old[3], old[4] + 1))
                        continue
            changed = True
            owner, _, method = q.rpartition(".")
            if bool(old) != bool(revised):
                name = method.split("#")[0]
                if owner in old_scopes and owner in new_scopes and name.startswith("__") and name.endswith("__"):
                    # Python can alter an inherited protocol without calling a
                    # same-named method on the baseline (e.g. implicit hashing).
                    dirty.add(owner)
                    out.dialect_subset_safe = False
                elif not old and owner in new_scopes and not name.startswith("_") and name.endswith("_sql"):
                    # Generator dispatch discovers methods by expression key;
                    # the old implementation may have used the generic renderer.
                    expr_mask = self.expression_keys().get(name[:-4], 0)
                    if expr_mask and (path == "sqlglot/generator.py" or path.startswith(
                            ("sqlglot/generators/", "sqlglot/dialects/"))):
                        out.broad |= expr_mask | (self.index["heap_observers"] & self.importers[path])
            if old and revised and old[1] == revised[1]:
                # Only function body changed: use statement-level granularity
                old_tree, old_lines = self.old_tree_and_lines(path)
                fn_old = _get_func_node(old_tree, q)
                fn_new = _get_func_node(new_tree_for_stmts, q)
                if fn_old is not None:
                    h0, h1 = _header_span(fn_old)
                    header_skip.update(range(h0, h1 + 1))
                if owner == "":
                    changed_symbols.add((mod_name, method.split("#")[0]))
                elif owner in old_scopes:
                    changed_symbols.add((f"{mod_name}.{owner}", method.split("#")[0]))
                # A local introduced on an untaken branch can rebind a name
                # throughout the frame. Include callers of unaffected lines too.
                frame_changed = not same_layout() and changes_frame_bindings(
                    self.old_code(path).get(q), code.get(q)
                )
                spans = None if frame_changed else _changed_stmt_spans(
                    fn_old, old_lines, new.splitlines(), fn_new
                )
                if out.query_spans is not None:
                    safe = body_spans(fn_old, fn_new, spans)
                    if safe:
                        out.query_spans.setdefault(path, []).extend(safe)
                    else:
                        out.query_spans = None
                l_hits = self.line_hits(path)
                if spans and l_hits:
                    stmt_mask = 0
                    for s_start, s_end in spans:
                        for ln in range(s_start, s_end + 1):
                            stmt_mask |= l_hits.get(ln, 0)
                    if stmt_mask and operand_safe:
                        stmt_mask = ((stmt_mask & self.index["heap_observers"]) |
                                     narrow(self.maps, path, fn_old, fn_new, stmt_mask, l_hits, same_layout))
                    out.direct |= stmt_mask
                else:
                    out.direct |= hits.get(q, 0)
            elif old:
                all_body_only = False
                # Def header changed or function deleted: all tests executing any line of function body
                out.direct |= hits.get(q, 0)
                if not revised:
                    l_hits = self.line_hits(path)
                    for ln in range(old[3], old[4] + 1):
                        out.direct |= l_hits.get(ln, 0)
            else:
                all_body_only = False
                same = self.index["by_name"].get(method.split("#")[0], 0)
                out.broad |= (same & runtime) or same
            if old and revised and old[1] != revised[1]:
                out.query_spans = None
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
        if out.query_spans and path in out.query_spans and not same_layout():
            out.query_spans = None
        if prioritize and (changed or dirty):
            out.priority |= self.changed_line_hits(path, new)
        refined = None
        if dirty and scope_edit and probe:
            status, rows = probe()
            if status == "broken":
                out.broken = True
                out.broad |= coverage | self.importers[path]
                return
            if status == "ok" and not changed:
                old_tree, _ = self.old_tree_and_lines(path)
                touched = _changed_old_lines(base_src, new)
                is_expr_cls = (
                    path.startswith("sqlglot/expressions/")
                    and self.expr_owners(path, dirty)
                    and all(f"{path}::{o}" in self.index.get("class_usage", {}) for o in dirty)
                )
                if not is_expr_cls:
                    dict_mask = self._dict_value_candidates(path, old_tree, new, rows)
                    if dict_mask is not None:
                        out.broad |= dict_mask
                        return
                if data_only(old_tree, new_tree_for_stmts, base_src, new, touched):
                    refined = self.state_refs.readers(rows)
                elif not rows and same_runtime_declarations(base_src, new):
                    refined = 0
                elif not rows and not any(self.indirect_hits.get(path, {}).get(n) or self.line_hits(path).get(n)
                                          for n in touched):
                    if data_only(old_tree, new_tree_for_stmts, base_src, new, touched, condition=True):
                        refined = 0
                elif single_edit and self._unused_added_imports(path, old_tree, new_tree_for_stmts, rows):
                    return
                if refined == 0 and not rows:
                    return
        limit = -1 if refined is None else refined
        import_rows = self.indirect_hits.get(path)
        if import_rows and (scope_edit or changed):
            base_bytes = (self.base / path).read_bytes()
            skip = set(header_skip) if (not scope_edit and changed and all_body_only) else set()
            usage = self.index.get("class_usage", {})
            if (dirty and "" not in dirty and usage and path.startswith("sqlglot/expressions/")
                    and all(f"{path}::{o}" in usage for o in dirty) and self.expr_owners(path, dirty)):
                # Expression class-body data (arg_types, flags) executes only while its own module is
                # imported; its effect reaches tests through the class, which the class-usage refinement
                # below selects. Function bodies run at import time keep the importer attribution.
                skip |= self.import_self_only.get(path, set()) - _function_lines(base_bytes, path)
            import_clean = None
            imp_mods = self.import_modules.get(path, {})
            imp_hits = self.import_hits.get(path, {})
            c_hits = self.cache_hits.get(path, {})
            for ln in _changed_old_lines(base_bytes, new):
                if ln in skip or ln in equivalent:
                    continue
                if (not scope_edit and changed and all_body_only and single_edit and probe
                        and (imp_hits.get(ln, 0) & limit) and ln in imp_mods
                        and all(m.startswith("sqlglot/") for m in imp_mods[ln])):
                    if import_clean is None:
                        status, rows = probe()
                        if status == "broken":
                            out.broken = True
                            out.broad |= coverage | self.importers[path]
                            return
                        import_clean = (status == "ok" and all((r[0], r[1]) in changed_symbols for r in rows))
                    if import_clean:
                        out.broad |= c_hits.get(ln, 0) & limit
                        continue
                out.broad |= import_rows.get(ln, 0) & limit
        if not dirty:
            return
        # A heap observer is a candidate if it executes the changed code (normal coverage rule),
        # or the change is module-level/class-level in a module it imports, or the module is broken.
        heap = self.index["heap_observers"] & self.importers[path]
        if not (path.startswith(("sqlglot/dialects/", "sqlglot/generators/")) or path == "sqlglot/generator.py"):
            # Heap observers only see allocations made while they run. Data edits elsewhere reach them
            # only through code they execute (runtime coverage of this module). test_generator_memory
            # builds every dialect's generator, so dialect/generator modules stay included.
            heap &= runtime
        out.broad |= heap
        if test_file:
            for owner in dirty:
                affected = self.class_tests[path + "::" + owner] if owner else 0
                out.broad |= affected or (self.file_tests[path] | coverage | self.importers[path])
            return
        class_usage = self.index.get("class_usage", {})
        # Class-usage refinement is only trusted for Expression node classes: they are created per
        # parse (so per-test instantiation records are reliable) and their class data (arg_types,
        # flags) is read through instances or the generator's `<key>_sql` dispatch. Other classes
        # (tokenizer/parser/generator/dialect cores) have cached or import-time instances whose use
        # is invisible to per-test records (corpus c1/5xwe: TokenizerCore.__slots__), so they keep the
        # conservative rule below.
        # Enum/helper classes (adv k1f1: DType member value) are used by reference, not instantiation.
        if ("" not in dirty and class_usage and path.startswith("sqlglot/expressions/")
                and self.expr_owners(path, dirty)):
            matched_all = True
            cls_mask = 0
            for owner in dirty:
                key = f"{path}::{owner}"
                if key in class_usage:
                    cls_mask |= class_usage[key]
                    cls_mask |= self.index["by_name"].get(owner.split(".")[-1].lower() + "_sql", 0)
                    # Tests that ran methods defined on this class (not the whole module: core.py
                    # defines the Expression base used by every test).
                    for q, mask in hits.items():
                        if q.startswith(owner + "."):
                            cls_mask |= mask
                else:
                    matched_all = False
                    break
            if matched_all:
                out.broad |= cls_mask & limit
                return
        dialect = Path(path).stem
        out.broad |= runtime & limit
        if path.startswith("sqlglot/") and dialect in self.dialects:
            out.broad |= (self.dialects[dialect] | self.index["lazy_imports"].get(path, 0)) & limit
        else:
            # Module-level non-class data changes keep the conservative rule.
            out.broad |= (coverage | self.importers[path]) & limit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    start = time.process_time()
    analyzer = Analyzer(args.dataset / "tests.json")
    print(f"map_load_cpu={time.process_time()-start:.4f}", file=sys.stderr)
    answers = json.loads((args.dataset / "answers.json").read_bytes())["commits"] if args.check else {}
    from runner import Session
    with Session() as state_session:
        return check_dataset(args, analyzer, answers, state_session)


def check_dataset(args, analyzer, answers, state_session):
    hits = failures = selected = 0
    for patch in sorted((args.dataset / "commits").glob("*.patch")):
        try:
            result = analyzer.analyze(patch.read_bytes().decode("utf-8", "surrogateescape"),
                                      session=lambda: state_session)
            candidates = sorted(result.selected())
        except Exception as e:
            print(f"fallback {patch.stem} {e!r}", file=sys.stderr)
            candidates = list(analyzer.ids)
        print(json.dumps({"commit": patch.stem, "candidates": candidates}))
        if args.check:
            failing = set(answers[patch.stem]["failing"])
            missing = sorted(failing - set(candidates))
            if missing:
                print(f"miss {patch.stem}: {[(n, analyzer.ids[n]) for n in missing]}", file=sys.stderr)
            failures += len(failing)
            selected += len(candidates)
            hits += len(failing) - len(missing)
    if args.check:
        print(f"recall={hits}/{failures} precision={hits}/{selected}", file=sys.stderr)
        return int(hits != failures)




class _MaskLambdas(ast.NodeTransformer):
    """Mask deferred bodies while keeping the definition-time defaults."""

    def __init__(self):
        self.bodies = []
        self.frames = []

    def visit_Lambda(self, node):
        self.bodies.append((node.lineno, node.end_lineno))
        bindings = frozenset(n.target.id for n in ast.walk(node.body) if isinstance(n, ast.NamedExpr))
        generator = any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(node.body))
        self.frames.append((bindings, generator))
        node.body = ast.copy_location(ast.Constant(value="<lambda>"), node.body)
        return self.generic_visit(node)


def _class_dict_nodes(tree):
    """Return ({(cls_qname, attr): ast.Dict}, ambiguous_keys) for direct class-body dict literals."""
    nodes = {}
    ambiguous = set()

    def walk(node, prefix=()):
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
                            key = (cname, targets[0].id)
                            if key in nodes:
                                ambiguous.add(key)
                            nodes[key] = item.value
                walk(stmt, (*prefix, stmt.name))

    walk(tree)
    return nodes, ambiguous


def _function_lines(src_bytes, name):
    """Old-file line numbers inside any function or lambda (including its decorators/signature)."""
    lines = set()
    for node in ast.walk(ast.parse(src_bytes, name)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            lines.update(range(start, (node.end_lineno or node.lineno) + 1))
    return lines


def _changed_old_lines(old_bytes, new_bytes):
    """Old-file line numbers (1-based) touched by the edit; pure insertions map to both neighbours."""
    old = old_bytes.splitlines()
    new = new_bytes.splitlines() if isinstance(new_bytes, (bytes, bytearray)) else new_bytes.encode().splitlines()
    lines = set()
    start = 0
    shared = min(len(old), len(new))
    while start < shared and old[start] == new[start]:
        start += 1
    a, b = len(old), len(new)
    while a > start and b > start and old[a - 1] == new[b - 1]:
        a -= 1
        b -= 1
    edits = SequenceMatcher(None, old[start:a], new[start:b], autojunk=False)
    for op, i1, i2, _, _ in edits.get_opcodes():
        if op == "equal":
            continue
        i1 += start
        i2 += start
        if i1 == i2:
            lines.update((i1, i1 + 1))
        else:
            lines.update(range(i1 + 1, i2 + 1))
    return lines


def _resolve_import_from(path, node):
    if node.level == 0:
        return node.module or ""
    parts = path[:-3].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    up = node.level - 1
    pkg = parts[:-up] if up else parts
    return ".".join([*pkg, *([node.module] if node.module else [])])


if __name__ == "__main__":
    raise SystemExit(main())
