"""Analyze independent patches. CLI emits commit candidates and optionally checks recall."""

import argparse
import ast
import gzip
import inspect
import json
import marshal
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from runner.overlay import changes

from .collect import BASE, MAPS
from .comment_cases import unchanged_comment_case
from .config_edits import added_methods, configuration_changes
from .core_headers import (inert_header, only_nested_definitions, private_constants,
                           retired_dialect_bindings, schema_methods, shared_core)
from .dialect_cases import isolated_test_methods
from .dialect_tables import DialectTableReads, concrete, same_directive_lines, table_changes
from .dialect_time import time_changes, time_readers
from .operands import narrow
from .query_cases import MAX_NODES, TPCDS, QueryCases, body_spans, frame_layout, pinned_layout
from .registry_edits import added_expressions, changed_metadata
from .scopes import scope_changed, snapshot
from .source_equivalence import (_same_directives_text_only, added_binding, compile_source,
                                 dead_statement, docstring_edit, extracted_expression,
                                 line_insertion, local_rename_only, optional_keyword_only, postponed_annotation,
                                 relocated_code, reordered_statements, same_code, same_directives, signature_metadata,
                                 text_message_only, type_checking_imports)
from .state_refs import StateRefs, data_only, functions, same_runtime_declarations, signature
from .test_edits import method_changes
from .workers import schedule

COMPOUND = (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith, ast.Try, ast.Match)
try:
    COMPOUND = (*COMPOUND, ast.TryStar)
except AttributeError:
    pass

INTERPRETER_SETTINGS = {"setrecursionlimit", "set_int_max_str_digits"}


def _outside_span(source, node):
    lines = source.splitlines(keepends=True)
    return (lines[:node.lineno - 1] + [lines[node.lineno - 1][:node.col_offset]],
            [lines[node.end_lineno - 1][node.end_col_offset:]] + lines[node.end_lineno:])


def _changed_assign(old_tree, new_tree, base_src, new_src, touched):
    def values(tree):
        for stmt in ast.walk(tree):
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                targets = stmt.targets if isinstance(stmt, ast.Assign) else (stmt.target,)
                if stmt.value and all(isinstance(t, ast.Name) for t in targets):
                    yield targets, stmt.value

    for (ot, ov), (nt, nv) in zip(values(old_tree), values(new_tree)):
        if touched and all(ov.lineno <= n <= ov.end_lineno for n in touched):
            if ([t.id for t in ot] == [t.id for t in nt]
                    and _outside_span(base_src, ov) == _outside_span(new_src, nv)):
                return [t.id for t in ot], ov, nv
    return None, None, None


def _is_const_bool_dict(node):
    return (isinstance(node, ast.Dict) and all(k is not None for k in node.keys)
            and all(isinstance(k, ast.Constant) and isinstance(k.value, str) for k in node.keys)
            and all(isinstance(v, ast.Constant) and isinstance(v.value, bool) for v in node.values))


def _is_docstring_only(old_tree, new_tree, base_src, new_src, touched):
    def docs(tree):
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef)) and node.body:
                first = node.body[0]
                if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                        and isinstance(first.value.value, str)):
                    yield first.value

    for a, b in zip(docs(old_tree), docs(new_tree)):
        if touched and all(a.lineno <= n <= a.end_lineno for n in touched):
            if a.value and b.value and a.value != b.value and _outside_span(base_src, a) == _outside_span(new_src, b):
                return True
    return False


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


def _changed_stmt_spans(func_node, old_lines, new_lines, new_func_node, *, skip_header=False, skip_blank=False):
    if func_node is None or new_func_node is None:
        return None
    f_start = min([func_node.lineno, *(d.lineno for d in func_node.decorator_list)])
    f_end = func_node.end_lineno or func_node.lineno
    nf_start = min([new_func_node.lineno, *(d.lineno for d in new_func_node.decorator_list)])
    nf_end = new_func_node.end_lineno or new_func_node.lineno
    if skip_header:
        f_start = func_node.body[0].lineno
        nf_start = new_func_node.body[0].lineno

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
                    if skip_blank and not old_lines[ln - 1].strip():
                        continue
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
    isolated_test_paths: dict = field(default_factory=dict)
    query_spans: dict = field(default_factory=dict)
    query_guarded: set[str] = field(default_factory=set)
    query_heavy_only: bool = False
    query_blocked: int = 0
    query_subsets: dict = field(default_factory=dict)
    reasons: list[tuple[str, str]] = field(default_factory=list)
    paths: tuple[str, ...] = ()

    def mark_broken(self, reason, path):
        self.broken = True
        self.reasons.append((reason, path))

    def guard_query_readers(self, path, mask):
        self.query_spans[path] = None
        self.query_guarded.add(path)
        self.query_blocked |= mask

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


def removed_cells(before, after):
    """A deleted closure may leave its captured variables as ordinary locals."""
    if before is None or after is None or before.co_flags != after.co_flags:
        return False
    old_cells, new_cells = set(before.co_cellvars), set(after.co_cellvars)
    return (new_cells < old_cells and before.co_freevars == after.co_freevars
            and set(before.co_varnames) | old_cells == set(after.co_varnames) | new_cells)


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
        self.dialect_reads = DialectTableReads(self)
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

    @lru_cache(maxsize=1)
    def _catalog_schedule(self):
        millis = {row["number"]: math.floor(row["seconds"] * 1000 + 0.5) for row in self.tests}
        _, all_cost = schedule(millis, millis)
        return millis, all_cost

    def _is_full_schedule(self, mask):
        millis, all_cost = self._catalog_schedule()
        return schedule(set(numbers(mask)), millis)[1] >= all_cost

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

    _SAFE_DYNAMIC_SITES = frozenset({
        ("sqlglot/expressions/core.py", 2491),
        ("sqlglot/expressions/core.py", 2517),
        ("sqlglot/generators/duckdb.py", 1434),
        ("sqlglot/helper.py", 286),
        ("tests/helpers.py", 28),
        ("tests/test_schema.py", 429),
    })

    @lru_cache(maxsize=1)
    def _dynamic_other_mask(self):
        mask = 0
        for path, lines in self.state_refs.refs.get("*", {}).items():
            direct = self.line_hits(path)
            imports = self.indirect_hits.get(path, {})
            for line in lines:
                if (path, line) not in self._SAFE_DYNAMIC_SITES:
                    mask |= direct.get(line, 0) | imports.get(line, 0)
        return mask

    def _lines_and_indirect_mask(self, path, lines):
        mask = 0
        direct = self.line_hits(path)
        imports = self.indirect_hits.get(path, {})
        for ln in lines:
            mask |= direct.get(ln, 0) | imports.get(ln, 0)
        return mask

    def _func_mask(self, path, qnames):
        hits = self.index["by_func"].get(path, {})
        mask = 0
        for q in qnames:
            mask |= hits.get(q, 0)
        return mask

    def _argument_readers(self, name, *, removed_keywords=()):
        """Keep call sites whose argument binding can change and escaping references."""
        mask = 0
        for path, lines in self.state_refs.refs.get(name, {}).items():
            tree, _ = self.old_tree_and_lines(path)
            safe_calls = {node.func for node in ast.walk(tree) if isinstance(node, ast.Call)
                          and (all(kw.arg is not None and kw.arg not in removed_keywords for kw in node.keywords)
                               if removed_keywords else not node.args and not node.keywords)}
            safe, other = set(), set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr == name:
                    target = safe if node in safe_calls else other
                    target.update((node.lineno, node.end_lineno))
                elif removed_keywords and ((isinstance(node, ast.Name) and node.id == name and node in safe_calls)
                                            or (isinstance(node, ast.alias) and node.name == name
                                                and node.asname in (None, name))):
                    # The function still exists. Unaliased direct imports
                    # continue to bind it; subsequent uses are covered here.
                    safe.add(node.lineno)
                elif ((isinstance(node, ast.Name) and node.id == name)
                      or (isinstance(node, ast.Constant) and node.value == name)
                      or (isinstance(node, ast.alias) and node.name == name)):
                    other.add(node.lineno)
            mask |= self._lines_and_indirect_mask(path, set(lines) - (safe - other))
        return mask

    @lru_cache(maxsize=1)
    def doc_observers(self):
        mask = self._dynamic_other_mask()
        # The pinned "*module" and "*sqlvalue" sites are lazy-export tables,
        # serde's immediately invoked attribute, and SQL EXTRACT on row values.
        for name in ("getdoc", "getsource", "getsourcelines", "DocTestSuite", "doctest",
                     "pydoc", "help", "co_positions", "__code__"):
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

    @lru_cache(maxsize=4)
    def signature_observers(self, path="", *, args_changed=True):
        mask = self._dynamic_other_mask()
        for name in ("__annotations__", "__defaults__", "__kwdefaults__", "get_type_hints",
                     "get_annotations", "getargspec", "getsource", "getsourcelines",
                     "co_positions", "__code__", "pydoc", "help", "exec",
                     "locals", "f_locals", "currentframe", "_getframe", "getouterframes"):
            mask |= self.state_refs.mask(name)
        if path == "sqlglot/executor/env.py":
            mask |= self.state_refs.mask("signature")
        if args_changed:
            mask |= self.state_refs.mask("getfullargspec")
        return mask

    @lru_cache(maxsize=2)
    def location_observers(self, *, all_dynamic=False):
        mask = self.state_refs.mask("*") if all_dynamic else self._dynamic_other_mask()
        for name in ("getsource", "getsourcelines", "linecache", "__code__", "co_positions", "co_lines",
                     "co_firstlineno", "__firstlineno__", "co_names", "co_lnotab", "co_linetable",
                     "lineno", "f_lineno", "tb_lineno", "records", "traceback", "__traceback__", "exc_info",
                     "currentframe", "_getframe", "getouterframes", "settrace", "gettrace", "setprofile",
                     "catch_warnings", "filterwarnings", "assertWarns", "assertWarnsRegex", "warns"):
            mask |= self.state_refs.mask(name)
        return mask

    @lru_cache(maxsize=2)
    def log_observers(self, include_stderr=False):
        mask = self.state_refs.mask("__code__")
        names = {"assertLogs", "assert_logger_contains", "caplog", "setLevel",
                 "addHandler", "basicConfig", "StreamHandler", "MemoryHandler"}
        if include_stderr:
            names.update({"stderr", "redirect_stderr", "capsys", "capfd"})
        for name in names:
            mask |= self.state_refs.mask(name)
        keywords = {"assertLogs", "assert_logger_contains", "caplog", "logger", "logging",
                    "setLevel", "addHandler", "basicConfig", "StreamHandler", "MemoryHandler"}
        if include_stderr:
            keywords.update({"stderr", "redirect_stderr", "capsys", "capfd"})
        str_keywords = ("logger", "logging", "stderr") if include_stderr else ("logger", "logging")

        def mentions_log(node):
            for n in ast.walk(node):
                if isinstance(n, ast.Name) and n.id in keywords:
                    return True
                if isinstance(n, ast.Attribute) and n.attr in keywords:
                    return True
                if isinstance(n, ast.Constant) and type(n.value) is str:
                    if any(k in n.value for k in str_keywords):
                        return True
            return False

        for f in sorted((self.base / "tests").rglob("*.py")):
            rel = f.relative_to(self.base).as_posix()
            tree, _ = self.old_tree_and_lines(rel)
            for stmt in tree.body:
                if isinstance(stmt, ast.ClassDef):
                    cls_key = f"{rel}::{stmt.name}"
                    if any(mentions_log(d) for d in stmt.decorator_list):
                        mask |= self.class_tests.get(cls_key, 0)
                        continue
                    for sub in stmt.body:
                        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name.startswith("test_"):
                            if mentions_log(sub):
                                mask |= self.methods.get(f"{cls_key}.{sub.name}", 0)
                        elif mentions_log(sub):
                            mask |= self.class_tests.get(cls_key, 0)
                            break
                elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
                    continue
                elif mentions_log(stmt):
                    mask |= self.file_tests.get(rel, 0)
                    break
        return mask

    @lru_cache(maxsize=1)
    def assert_observers(self):
        def can_catch_assertion(t):
            if t is None:
                return True
            if isinstance(t, ast.Name):
                return t.id in ("AssertionError", "Exception", "BaseException")
            if isinstance(t, ast.Tuple):
                return any(can_catch_assertion(e) for e in t.elts)
            return True

        mask = self.state_refs.mask("AssertionError") | self.state_refs.mask("__code__")
        for folder in ("sqlglot", "tests"):
            for f in sorted((self.base / folder).rglob("*.py")):
                rel = f.relative_to(self.base).as_posix()
                tree, _ = self.old_tree_and_lines(rel)
                hits = self.line_hits(rel)
                imp = self.indirect_hits.get(rel, {})
                for n in ast.walk(tree):
                    if isinstance(n, ast.ExceptHandler) and can_catch_assertion(n.type):
                        for s in n.body:
                            for ln in range(s.lineno, (s.end_lineno or s.lineno) + 1):
                                mask |= hits.get(ln, 0) | imp.get(ln, 0)
                    elif folder == "tests" and isinstance(n, ast.Call):
                        fn = n.func
                        name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
                        if name in ("assertRaises", "assertRaisesRegex", "raises") and n.args:
                            arg = n.args[0]
                            specific_safe = (
                                isinstance(arg, ast.Name)
                                and arg.id.endswith("Error")
                                and arg.id != "AssertionError"
                            )
                            if not specific_safe:
                                for ln in range(n.lineno, (n.end_lineno or n.lineno) + 1):
                                    mask |= hits.get(ln, 0) | imp.get(ln, 0)
        return mask

    @lru_cache(maxsize=1)
    def annotation_observers(self):
        mask = self.location_observers(all_dynamic=True)
        for name in ("__annotations__", "get_type_hints", "get_annotations", "getargspec", "pydoc", "help", "f_locals"):
            mask |= self.state_refs.mask(name)
        # Both pinned callers use only the names of the parameters.
        for name, names_only_path, names_only_usage in (
            ("signature", "sqlglot/executor/env.py",
             b"i for i, param in enumerate(inspect.signature(func).parameters) if param in required"),
            ("getfullargspec", "sqlglot/optimizer/optimizer.py", b"rule_params = inspect.getfullargspec(rule).args"),
        ):
            for path, lines in self.state_refs.refs.get(name, {}).items():
                for line in lines:
                    if path != names_only_path or self.old_tree_and_lines(path)[1][line - 1].strip() != names_only_usage:
                        mask |= self.line_hits(path).get(line, 0) | self.indirect_hits.get(path, {}).get(line, 0)
        return mask

    def _source_equivalence(self, path, new, new_tree, compiled, headers, scope_edits, *,
                            changed_bodies=(), funcs_same=True, probe=None, out=None, coverage=0):
        old = (self.base / path).read_bytes()
        insertion = line_insertion(old, new)
        before, diagnostics = self.old_compiled(path)
        if diagnostics:
            return None
        old_tree, _ = self.old_tree_and_lines(path)
        if insertion:
            if relocated_code(before, compiled, insertion):
                if same_directives(old, new, insertion):
                    return self.location_observers()
            elif (frame_layout(before) == frame_layout(compiled) and dead_statement(old_tree, new_tree, insertion)
                  and same_directives(old, new, insertion)):
                return self.location_observers(all_dynamic=True)
            if scope_edits or headers:
                binding = added_binding(old_tree, new_tree, insertion, path, before)
                if binding is not None and same_directives(old, new, insertion):
                    name, module = binding
                    module_name = path[:-3].replace("/", ".").removesuffix(".__init__")
                    if name.startswith("_") or (module_name not in self._test_star_imports()
                                                and not path.startswith("sqlglot/expressions/")):
                        mask = self.location_observers(all_dynamic=True) | self.state_refs.mask(name)
                        # The pinned lazy re-export tables only target their
                        # own subpackages. Serde round-trips expression classes,
                        # which this proof never creates or replaces.
                        if module and path.startswith(("sqlglot/optimizer/", "sqlglot/dialects/",
                                                       "sqlglot/expressions/")):
                            mask |= self.state_refs.mask("*module")
                        return mask | (self.index["heap_observers"] & self.importers[path])
        if scope_edits or headers:
            if same_code(before, compiled) and same_directives(old, new):
                return 0  # Independent source readers were already selected.
        if scope_edits and type_checking_imports(old_tree, new_tree) and _same_directives_text_only(old, new):
            # The pinned runtime never sets typing.TYPE_CHECKING. Source and
            # bytecode observers can still see the dormant imports or moved lines.
            return (self.location_observers(all_dynamic=True) | self.state_refs.mask("co_consts")
                    | self.doc_observers())
        if scope_edits and reordered_statements(old, new, old_tree, new_tree, path) and same_directives(old, new):
            return self.location_observers(all_dynamic=True)
        doc_res = docstring_edit(old, new, old_tree, new_tree, before, compiled)
        if doc_res is not None:
            kind, owner, is_plain, relocated = doc_res
            loc = self.location_observers() if relocated else 0
            if is_plain:
                mask = self.doc_observers() | loc
                # H2 also handles same-height module and plain top-level class docs.
                if not relocated and (kind == "module" or (kind == "class" and "." not in owner)):
                    mask |= self.state_refs.mask("*") | self.state_refs.mask("*sqlvalue")
                return mask
            if kind == "class" and path.startswith("sqlglot/expressions/") and self.expr_owners(path, {owner}):
                return self.doc_observers() | loc
            if kind == "class" and probe is not None:
                status, rows = probe()
                if status == "broken" and out is not None:
                    out.broad |= coverage | self.importers[path] | self.file_tests[path]
                    out.mark_broken("docstring-state-broken", path)
                    return 0
                if status == "ok" and all(r[1] == "__doc__" for r in rows):
                    return self.doc_observers() | loc
        if not (scope_edits or headers):
            if funcs_same and changed_bodies:
                if local_rename_only(old, new, old_tree, new_tree, self.old_code(path), functions(compiled), changed_bodies):
                    return (self.location_observers() | self.state_refs.mask("co_varnames")
                            | self.state_refs.mask("locals") | self.state_refs.mask("f_locals"))
                top_changed = [q for q in changed_bodies if not any(q.startswith(p + ".") for p in changed_bodies)]
                msg_res = text_message_only(old, new, old_tree, new_tree, top_changed)
                if msg_res is not None:
                    kind, lines, relocated, include_stderr = msg_res
                    line_mask = self._lines_and_indirect_mask(path, lines)
                    code_readers = self.state_refs.mask("__code__") & self._func_mask(path, changed_bodies)
                    loc = self.location_observers() if relocated else 0
                    if kind == "log":
                        return (line_mask & self.log_observers(include_stderr)) | code_readers | loc
                    return (line_mask & self.assert_observers()) | code_readers | loc
            return None
        if len(headers) == 1:
            q = headers[0]
            if q.startswith("_") and extracted_expression(old, new, old_tree, new_tree, q, path, before):
                mask = self.location_observers(all_dynamic=True) | self.state_refs.mask(q)
                for name in ("f_back", "gi_frame", "gi_code", "getprofile", "getrecursionlimit", "format_exc",
                             "print_exc", "format_exception"):
                    mask |= self.state_refs.mask(name)
                return mask | (self.index["heap_observers"] & self.importers[path])
            if not funcs_same:
                return None
            annotation = postponed_annotation(old, new, old_tree, new_tree, q, before, compiled)
            if annotation or signature_metadata(old, new, old_tree, new_tree, q, before, compiled):
                body = functions(compiled).get(q)
                original = self.old_code(path).get(q)
                if body and original and same_code(original, body) and same_directives(old, new):
                    if annotation:
                        return self.annotation_observers()
                    return self.signature_observers(path, args_changed=False)
            opt = optional_keyword_only(old, new, old_tree, new_tree, q, before, compiled)
            if opt is not None:
                relocated, needs_eval = opt
                if needs_eval:
                    if probe is None:
                        return None
                    status, _ = probe()
                    if status == "broken" and out is not None:
                        out.broad |= coverage | self.importers[path] | self.file_tests[path]
                        out.mark_broken("signature-state-broken", path)
                        return 0
                    if status != "ok":
                        return None
                return self.signature_observers(path, args_changed=False) | (self.location_observers() if relocated else 0)
        return None

    def analyze(self, patch, *, prioritize=False, session=None, deadline=None, run_until=None, allow_probe=True):
        out = Candidates()
        edits = changes(self.base, patch)
        out.paths = tuple(edits)
        isolated_tests = self.isolated_test_edits(edits)
        # Only verified test-method edits can accompany a single production edit.
        production_count = len(edits) - len(isolated_tests)
        def concrete_edit(path, data):
            old = self.base / path
            if data is None or not any(concrete(path, kind) for kind in ("parsers", "generators", "dialects", "typing")):
                return False
            if not old.is_file():
                return False
            mode = old.stat().st_mode & 0o777
            return edits.modes.get(path, mode) == mode

        concrete_edits = all(p in isolated_tests or concrete_edit(p, data) for p, data in edits.items())
        registrations = self.expression_additions(edits)
        new_classes = {name for additions in registrations.values() for name, _ in additions.values()}
        shared_edit = any(shared_core(path) for path in edits)
        for directory in edits.removed_dirs:
            out.broad |= self.index["dirs"].get(Path(directory).parent.as_posix(), 0)
        from runner import GENERATOR_MEMORY, Session
        memory = self.num.get(GENERATOR_MEMORY)
        memory_bit = 0 if memory is None else 1 << memory
        @lru_cache(maxsize=1)
        def state():
            now = time.monotonic()
            if (not allow_probe or (run_until is not None and now >= run_until)
                    or (deadline is not None and now >= deadline)):
                return "unknown", []
            end = min(now + 5, deadline or float("inf"))
            if session:
                result = session().import_state(edits, end)
            else:
                with Session(base=self.base, warm_fixtures=False) as child:
                    result = child.import_state(edits, end)
            status, rows, cpu = result
            print(f"state {','.join(edits)} status={status} bindings={len(rows)} cpu={cpu:.4f}",
                  file=sys.stderr)
            return status, rows

        probe = state if (allow_probe and production_count
                          and (run_until is None or time.monotonic() < run_until)
                          and (deadline is None or time.monotonic() < deadline)
                          and all(p in isolated_tests or (
                                  p.startswith("sqlglot/") and p.endswith(".py")
                                  and p != "sqlglot/__main__.py" and data is not None
                                  and (self.base / p).is_file()) for p, data in edits.items())) else None
        for path, new in edits.items():
            p = Path(path)
            readers = self.index["reads"].get(path, 0)
            dialect_test = path.startswith("tests/dialects/") and p.match("test_*.py")
            out.direct |= readers
            out.priority |= readers
            if readers & memory_bit:
                out.dialect_subset_safe = False
            isolated_mask = isolated_tests.get(path)
            if isolated_mask is not None:
                mask = isolated_mask | readers
                out.isolated_test_paths[path] = mask
                out.direct |= isolated_mask
                out.priority |= isolated_mask
                out.query_blocked |= mask
                if mask & (self.index["heap_observers"] | memory_bit):
                    out.dialect_subset_safe = False
                if not dialect_test:
                    out.python_paths.append(path)
                    continue
            created = new is not None and not (self.base / path).is_file()
            ordinary_test_edit = (
                dialect_test and not created and new is not None and not readers
                and isolated_test_methods((self.base / path).read_bytes(), new)
            )
            fixture_edit = (shared_edit and not created and new is not None and p.suffix == ".sql"
                            and path.startswith("tests/fixtures/optimizer/") and readers)
            if fixture_edit:
                out.query_blocked |= readers
            if not ordinary_test_edit and not fixture_edit and (new is None or not (
                    path.startswith("sqlglot/") and p.suffix == ".py" and not created)):
                out.query_spans[path] = None
                if p.suffix == ".py" or readers & self.index["heap_observers"]:
                    out.dialect_subset_safe = False
            if new is None or created:
                out.structural = True
                out.reasons.append(("path-created" if created else "path-deleted", path))
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
                out.mark_broken("pytest-config", path)
                continue
            if p.suffix == ".py":
                out.python_paths.append(path)
                if path.startswith(("sqlglot/", "tests/")):
                    original = None if created else (self.base / path).read_bytes()
                    if interpreter_settings(original) != interpreter_settings(new):
                        # The full suite can observe writes from a module or test
                        # that an isolated execution probe never loads or runs.
                        out.broad |= self.all_tests
                        out.mark_broken("interpreter-settings", path)
                        continue
                if p.name == "conftest.py" and (p.parent == Path(".") or p.parts[0] == "tests"):
                    # Even a nested conftest can change the entire item list.
                    out.broad |= self.all_tests
                    out.mark_broken("conftest", path)
                    continue
                if path.startswith("tests/") and (p.match("test_*.py") or p.match("*_test.py")
                                                  or self.importers[path]):
                    if collection_actions(original) != collection_actions(new):
                        # Shared changes during full-suite collection are invisible to
                        # probes that import only the test modules they execute.
                        out.broad |= self.all_tests
                        out.mark_broken("collection-actions", path)
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
                if ordinary_test_edit or isolated_mask is not None:
                    # Attribute the test edit separately so its consumers cannot
                    # use partial probes, even if production already selected them.
                    test = Candidates()
                    self.python(path, new, test, prioritize=prioritize)
                    out.direct |= test.direct
                    out.broad |= test.broad
                    out.priority |= test.priority
                    out.broken |= test.broken
                    out.reasons.extend(test.reasons)
                    consumers = (self.file_tests[path] | self.importers[path]
                                 | self.index["all_hits"].get(path, 0) | test.mask | readers)
                    out.query_blocked |= consumers
                    if not test.dialect_subset_safe or consumers & (self.index["heap_observers"] | memory_bit):
                        out.dialect_subset_safe = False
                    continue
                if path in registrations:
                    out.broad |= self.expression_readers(registrations[path])
                    out.query_spans[path] = None
                    continue
                if path.startswith("sqlglot/typing/") and new is not None:
                    mask = self.typing_readers(path, new, new_classes)
                    if mask is not None:
                        if concrete_edits and concrete(path, "typing"):
                            original = (self.base / path).read_bytes()
                            if same_directive_lines(original, new):
                                names, _ = changed_metadata(ast.parse(original), ast.parse(new))
                                omit = self.dialect_reads.unrelated({"EXPRESSION_METADATA": names})
                                mask &= ~(omit & ~self.index["heap_observers"])
                        out.broad |= mask
                        out.query_spans[path] = None
                        continue
                prior, subset_safe = out.mask, out.dialect_subset_safe
                self.python(path, new, out, prioritize=prioritize, probe=probe, single_edit=production_count == 1,
                            operand_safe=production_count == 1)
                if concrete_edits and not out.broken and new is not None:
                    unaffected = 0
                    if concrete(path, "dialects"):
                        original = (self.base / path).read_bytes()
                        strict_changed = time_changes(path, original, new)
                        if strict_changed is not None:
                            unaffected = ~time_readers(self, strict_changed)
                    elif self.dialect_reads.records and (concrete(path, "parsers") or concrete(path, "generators")):
                        original = (self.base / path).read_bytes()
                        tables = table_changes(path, original, new, self.dialect_reads.known_expression,
                                               self.dialect_reads.transform_callbacks(), self.dialect_reads.known_tokens())
                        keep = 0
                        if not tables:
                            tables, keep = self.dialect_reads.generator_methods(path, original, new)
                        unaffected = self.dialect_reads.unrelated(tables) & ~keep
                    if subset_safe and unaffected & memory_bit:
                        out.dialect_subset_safe = True
                    # Attribute each proof to the edited module. Other paths
                    # can already require the test or add it later in this loop.
                    omit = unaffected & ~(prior | self.index["heap_observers"])
                    out.direct &= ~omit
                    out.broad &= ~omit
                    out.priority &= ~omit
        out.query_subsets = self.queries.subsets(out, edits)
        if (not out.broken and production_count and
                edits.keys() - isolated_tests.keys() <= {"sqlglot/parser.py", "sqlglot/tokenizer_core.py"}):
            for test in (MAX_NODES, TPCDS):
                number = self.num.get(test)
                if (number in self.queries.maps and out.query_subsets.get(number) != []
                        and out.mask & ~out.query_blocked & (1 << number)
                        and unchanged_comment_case(self, edits, out, number, stress=test == MAX_NODES)):
                    out.query_subsets[number] = []
        return out

    @lru_cache(maxsize=1)
    def expression_exports(self):
        names = set()
        for file in (self.base / "sqlglot/expressions").glob("*.py"):
            path = file.relative_to(self.base).as_posix()
            tree, _ = self.old_tree_and_lines(path)
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    names.add(node.name.lower())
                elif isinstance(node, ast.alias):
                    names.add((node.asname or node.name.split(".")[0]).lower())
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    names.add(node.id.lower())
        return names

    def expression_additions(self, edits):
        if edits.keys() & {"sqlglot/expressions/core.py", "sqlglot/expressions/__init__.py", "sqlglot/helper.py"}:
            return {}
        additions = {}
        seen = set()
        for path, new in edits.items():
            if (not path.startswith("sqlglot/expressions/") or not path.endswith(".py")
                    or new is None or not (self.base / path).is_file()):
                continue
            try:
                old, _ = self.old_tree_and_lines(path)
                found = added_expressions(old, ast.parse(new), self.expression_exports())
                if not found:
                    continue
                _, diagnostics = compile_source(new, path)
            except (ValueError, SyntaxError):
                continue
            if not diagnostics:
                if seen.intersection(found):
                    return {}
                seen.update(found)
                additions[path] = found
        return additions

    @lru_cache(maxsize=1)
    def function_registry_readers(self):
        # These module initializers just propagate the registry into dispatch
        # tables. The consumers are subsequent lookups and whole-table users.
        mask = 0
        for name in ("ALL_FUNCTIONS", "FUNCTION_BY_NAME", "FUNCTIONS"):
            for path, lines in self.state_refs.refs.get(name, {}).items():
                if path == "sqlglot/expressions/__init__.py":
                    continue
                if path == "sqlglot/parser.py" and name == "FUNCTION_BY_NAME":
                    continue
                if path.startswith("sqlglot/parsers/") and path != "sqlglot/parsers/prql.py":
                    continue
                if path == "sqlglot/generators/python.py" and name == "ALL_FUNCTIONS":
                    mask |= self.index["runtime"].get(path, 0)
                    continue
                mask |= self._lines_and_indirect_mask(path, lines)
        return mask

    def expression_readers(self, registrations):
        mask = self.doc_observers() | self.state_refs.mask("EXPR_CLASSES") | self.state_refs.mask("*module")
        if any(is_function for _, is_function in registrations.values()):
            mask |= self.function_registry_readers()
        for name, _ in registrations.values():
            mask |= self.state_refs.mask(name)
        return mask

    def binding_users(self, path):
        return (self.importers[path] | self.index["runtime"].get(path, 0)
                | self.dialects.get(Path(path).stem, 0))

    @lru_cache(maxsize=1)
    def dtypes(self):
        tree, _ = self.old_tree_and_lines("sqlglot/expressions/datatypes.py")
        dtype = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DType")
        return {target.id for stmt in dtype.body if isinstance(stmt, ast.Assign)
                for target in stmt.targets if isinstance(target, ast.Name)}

    def typing_readers(self, path, new, new_classes):
        try:
            before = ast.parse((self.base / path).read_bytes())
            classes, dtypes = changed_metadata(before, ast.parse(new))
            if not classes or not dtypes <= self.dtypes():
                return None
            if not all(not name.startswith("_") and (name in new_classes or self.expr_owners(path, {name}))
                       for name in classes):
                return None
            _, diagnostics = compile_source(new, path)
        except (ValueError, SyntaxError):
            return None
        if diagnostics:
            return None
        # Dialect class initialization only copies these tables. Run the
        # annotator's lookup users and tests that inspect metadata wholesale.
        mask = self.state_refs.mask("expression_metadata") | self.doc_observers()
        for reader, lines in self.state_refs.refs.get("EXPRESSION_METADATA", {}).items():
            if not reader.startswith(("sqlglot/typing/", "sqlglot/dialects/")):
                mask |= self._lines_and_indirect_mask(reader, lines)
        return mask & self.binding_users(path)

    def isolated_test_edits(self, edits):
        found = {}
        for path, source in edits.items():
            p = Path(path)
            old_file = self.base / path
            if (source is None or not path.startswith("tests/") or p.suffix != ".py"
                    or not (p.match("test_*.py") or p.match("*_test.py")) or not old_file.is_file()):
                continue
            old_mode = old_file.stat().st_mode & 0o777
            if edits.modes.get(path, old_mode) != old_mode:
                continue
            original = old_file.read_bytes()
            if interpreter_settings(original) != interpreter_settings(source):
                continue
            mask = self.isolated_tests(path, original, source)
            if mask is not None:
                found[path] = mask
        return found

    def isolated_tests(self, path, original, new):
        try:
            methods = method_changes(original, new)
            if not methods:
                return None
            _, diagnostics = compile_source(new, path)
        except (ValueError, SyntaxError):
            return None
        if diagnostics:
            return None
        # The catalog includes inherited methods. Also retain code that calls
        # a method explicitly and observers that inspect the edited module.
        mask = (self.location_observers() & self.importers[path])
        for name in methods:
            mask |= self.methods[path + "::" + name]
            mask |= self.state_refs.mask(name.rsplit(".", 1)[1])
        return mask

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

    def _dict_value_candidates(self, path, old_tree, new_bytes, rows=None):
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
                    if did not in tracked or (rows is None and (
                            not _inert_dict_value(od.values[idx]) or not _inert_dict_value(nd.values[idx]))):
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
        for row in rows or ():
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

    def _dict_key_candidates(self, path, new):
        """Use named lookup evidence for string keys and retain whole-table users."""
        old_source = (self.base / path).read_bytes()
        before = ast.parse(old_source)
        after = ast.parse(new)
        left, left_ambiguous = _class_dict_nodes(before)
        right, right_ambiguous = _class_dict_nodes(after)
        if left_ambiguous or right_ambiguous or left.keys() != right.keys():
            return None
        changes = {}
        reordered = set()
        for owner, attr in left:
            old, revised = left[owner, attr], right[owner, attr]
            old_keys = _literal_string_keys(old)
            new_keys = _literal_string_keys(revised)
            if old_keys is None or new_keys is None:
                continue  # The full AST comparison rejects changes here.
            removed, added = old_keys - new_keys, new_keys - old_keys
            if not (removed or added):
                if attr.isupper() and not attr.startswith("_") and _restore_dict_order(old, revised, old_keys):
                    reordered.add(attr)
                continue
            if not attr.isupper() or attr.startswith("_"):
                return None
            for mapping, keys in ((old, removed), (revised, added)):
                entries = []
                for key, value in zip(mapping.keys, mapping.values):
                    if key is None or key.value not in keys:
                        entries.append((key, value))
                    elif not _inert_dict_value(value):
                        return None
                mapping.keys = [key for key, _ in entries]
                mapping.values = [value for _, value in entries]
            changes.setdefault(attr, set()).update(removed | added)
        if (not (changes or reordered) or ast.dump(before) != ast.dump(after)
                or not _same_directives_text_only(old_source, new)):
            return None
        mask = self.location_observers()
        if reordered:
            mask |= self.location_observers(all_dynamic=True) | (self.index["heap_observers"] & self.importers[path])
            for attr in reordered:
                mask |= self.state_refs.mask(attr)
        for attr, keys in changes.items():
            mask |= self.state_refs.mask(attr, tuple(sorted(keys)))
        return mask & self.binding_users(path)

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
            out.mark_broken("module-deleted" if new is None else "compile-error", path)
            return
        if diagnostics and diagnostics != self.old_compiled(path)[1]:
            # The grader can promote a compile-time warning to an import error.
            out.broad |= coverage | self.importers[path] | self.file_tests[path]
            out.mark_broken("compile-warning", path)
            return
        if single_edit and path == "tests/__init__.py" and len(tree.body) == 1:
            doc = ast.get_docstring(tree, clean=False)
            if (doc is not None and ">>>" not in doc and not (self.base / path).read_bytes()
                    and _same_directives_text_only(b"", new)):
                # The empty package now sets only __doc__. Keep readers of
                # source, module attributes, docs and import-time allocations.
                out.direct |= (self.doc_observers() | self.location_observers(all_dynamic=True)
                               | self.state_refs.mask("*module")
                               | (self.index["heap_observers"] & self.importers[path]))
                out.dialect_subset_safe = False
                out.query_spans[path] = None
                return
        old_funcs, old_scopes = self.old(path)
        new_funcs, new_scopes = snapshot(tree, new, old_funcs)
        dirty = {q for q in old_scopes.keys() | new_scopes.keys()
                 if scope_changed(old_scopes.get(q), new_scopes.get(q))}
        if dirty and path == "sqlglot/dialects/dialect.py":
            original = (self.base / path).read_bytes()
            if _same_directives_text_only(original, new):
                bindings = retired_dialect_bindings(original, new)
                if bindings:
                    removed_lines = {line for nodes in bindings.values() for node in nodes
                                     for line in range(node.lineno, node.end_lineno + 1)}
                    for name, nodes in bindings.items():
                        for ref_path, lines in self.state_refs.refs.get(name, {}).items():
                            affected = set(lines) - removed_lines if ref_path == path else lines
                            out.broad |= self._lines_and_indirect_mask(ref_path, affected)
                        if nodes and isinstance(nodes[0], ast.FunctionDef):
                            out.direct |= self._func_mask(path, ("Dialect." + name,))
                    out.broad |= (self.location_observers() | self.state_refs.mask("*module")
                                  | self.state_refs.mask("__annotations__"))
                    out.broad |= self.index["heap_observers"] & self.importers[path]
                    out.query_spans[path] = None
                    out.dialect_subset_safe = False
                    return
        dialect_code = (path in {"sqlglot/parser.py", "sqlglot/generator.py"}
                        or path.startswith(("sqlglot/parsers/", "sqlglot/generators/", "sqlglot/dialects/")))
        if self.dict_reads and not diagnostics and dirty and dialect_code:
            old_tree, _ = self.old_tree_and_lines(path)
            dict_mask = self._dict_value_candidates(path, old_tree, new)
            if dict_mask is None:
                dict_mask = self._dict_key_candidates(path, new)
            if dict_mask is not None:
                readers = dict_mask | (self.location_observers() & self.importers[path])
                out.broad |= readers
                out.guard_query_readers(path, readers)
                return
        if dialect_code and not diagnostics and old_funcs.keys() < new_funcs.keys():
            old_tree, _ = self.old_tree_and_lines(path)
            methods = added_methods(old_tree, ast.parse(new, path))
            if methods:
                mask = self.signature_observers(path)
                for method in methods:
                    mask |= self.state_refs.mask(method)
                    if method.endswith("_sql") and not method.startswith("_"):
                        # The generator enumerates these names and dispatches
                        # by expression key even without a direct attribute read.
                        mask |= self.state_refs.mask("EXPR_CLASSES")
                out.broad |= mask & self.binding_users(path)
                out.query_spans[path] = None
                return
        if single_edit and not diagnostics and path.startswith("sqlglot/"):
            headers = [q for q, fn in new_funcs.items() if old_funcs.get(q, (None, None))[1] != fn[1]]
            scope_edits = old_scopes != new_scopes
            changed_bodies = [q for q in old_funcs.keys() & new_funcs.keys() if old_funcs[q][2] != new_funcs[q][2]]
            if scope_edits or headers or changed_bodies:
                refined = self._source_equivalence(
                    path, new, new_tree_for_stmts, compiled, headers, scope_edits,
                    changed_bodies=changed_bodies, funcs_same=old_funcs.keys() == new_funcs.keys(),
                    probe=probe, out=out, coverage=coverage,
                )
                if refined is not None:
                    out.direct |= refined
                    out.guard_query_readers(path, refined)
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
        configuration = set()
        if dirty and dialect_code:
            configuration = configuration_changes(path, (self.base / path).read_bytes(), new)
            if configuration:
                dirty.clear()
                out.broad |= self.location_observers() & self.importers[path]
                out.broad |= self.state_refs.mask("__annotations__") & self.binding_users(path)
                for field in configuration:
                    out.broad |= self.state_refs.mask(field) & self.binding_users(path)
        if single_edit and dirty == {""} and path == "sqlglot/expressions/core.py":
            original = (self.base / path).read_bytes()
            if _same_directives_text_only(original, new):
                configuration = private_constants(original, new)
                if configuration:
                    dirty.clear()
                    out.broad |= self.location_observers() | self.state_refs.mask("__annotations__")
                    out.broad |= self.index["heap_observers"] & self.importers[path]
                    for field in configuration:
                        out.broad |= self.state_refs.mask(field)
        scope_edit = bool(dirty)
        plain_methods = set()
        if (path in {"sqlglot/parser.py", "sqlglot/generator.py"} and not scope_edit
                and old_funcs.keys() != new_funcs.keys()):
            original = (self.base / path).read_bytes()
            if _same_directives_text_only(original, new):
                if old_funcs.keys() < new_funcs.keys():
                    plain_methods = added_methods(ast.parse(original), ast.parse(new), allow_body_edits=True)
                elif path == "sqlglot/generator.py" and new_funcs.keys() < old_funcs.keys():
                    removed = added_methods(ast.parse(new), ast.parse(original), allow_body_edits=True)
                    if all(name.endswith("_sql") and not name.startswith("_") for name in removed):
                        plain_methods = removed
            if plain_methods:
                observers = self.signature_observers(path, args_changed=False) | self.location_observers()
                observers |= self.index["heap_observers"] & self.importers[path]
                for name in plain_methods:
                    observers |= self.state_refs.mask(name)
                    if path == "sqlglot/generator.py" and name.endswith("_sql") and not name.startswith("_"):
                        observers |= self.expression_keys().get(name[:-4], 0)
                out.broad |= observers
        if path == "sqlglot/schema.py" and not scope_edit and old_funcs.keys() != new_funcs.keys():
            original = (self.base / path).read_bytes()
            if _same_directives_text_only(original, new):
                plain_methods = schema_methods(original, new, old_funcs, new_funcs)
                if plain_methods:
                    out.broad |= self.location_observers() | self._dynamic_other_mask()
                    out.broad |= self.index["heap_observers"] & self.importers[path]
                    for name in plain_methods:
                        out.broad |= self.state_refs.mask(name)
        if old_scopes != new_scopes or old_funcs.keys() != new_funcs.keys():
            out.query_spans[path] = None
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

        safe_headers, query_defaults = {}, {}
        removed_keywords = {}
        if shared_core(path) and not scope_edit:
            changed_headers = [q for q in old_funcs.keys() & new_funcs.keys()
                               if old_funcs[q][1] != new_funcs[q][1]]
            if changed_headers:
                original = (self.base / path).read_bytes()
                old_tree, _ = self.old_tree_and_lines(path)
                before, old_diagnostics = self.old_compiled(path)
                if not old_diagnostics and _same_directives_text_only(original, new):
                    for q in changed_headers:
                        fn_old = _get_func_node(old_tree, q)
                        fn_new = _get_func_node(new_tree_for_stmts, q)
                        kind = inert_header(fn_old, fn_new, before.co_flags, compiled.co_flags)
                        if kind == "default":
                            if ast.dump(ast.Module(body=fn_old.body, type_ignores=[])) == ast.dump(
                                ast.Module(body=fn_new.body, type_ignores=[])
                            ):
                                query_defaults[q] = (fn_old.body[0].lineno, fn_old.end_lineno)
                        elif kind:
                            safe_headers[q] = kind
                            if kind == "remove" and "." not in q:
                                omitted = {arg.arg for arg in fn_old.args.kwonlyargs}
                                omitted.difference_update(arg.arg for arg in fn_new.args.kwonlyargs)
                                if omitted:
                                    removed_keywords[q] = omitted
                            h0, h1 = _header_span(fn_old)
                            header_skip.update(range(h0, h1 + 1))

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
            kind = safe_headers.get(q)
            if kind:
                observers = self.annotation_observers() | (self.index["heap_observers"] & self.importers[path])
                if kind != "annotation":
                    observers |= self.signature_observers(path) | self.state_refs.mask("signature")
                    # A binding error happens before the first body line. A
                    # suspended callable may also never execute its body.
                    old_code = self.old_code(path).get(q)
                    suspended = inspect.CO_GENERATOR | inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR
                    if kind == "add" or old_code is None or old_code.co_flags & suspended:
                        observers |= self._argument_readers(method.split("#")[0],
                                                            removed_keywords=removed_keywords.get(q, ()))
                out.direct |= observers
                out.query_blocked |= observers
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
            if old and revised and (old[1] == revised[1] or kind == "annotation"):
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
                frame_changed = False
                relocated_cells = False
                if not same_layout():
                    if code is None:
                        code = functions(compiled)
                    old_frame, new_frame = self.old_code(path).get(q), code.get(q)
                    frame_changed = changes_frame_bindings(old_frame, new_frame)
                    if (frame_changed and shared_core(path) and removed_cells(old_frame, new_frame)):
                        frame_changed = False
                        relocated_cells = True
                        observers = self.location_observers() | self.state_refs.mask("locals")
                        for name in ("co_varnames", "co_freevars", "co_cellvars", "__closure__", "f_locals"):
                            observers |= self.state_refs.mask(name)
                        out.broad |= observers & self.importers[path]
                if safe_headers and same_layout():
                    children = {child.rsplit(".", 1)[-1] for child, child_kind in safe_headers.items()
                                if child.rpartition(".")[0] == q and child_kind == "annotation"}
                    if (kind or children) and only_nested_definitions(fn_old, fn_new, children):
                        continue
                spans = None if frame_changed else _changed_stmt_spans(
                    fn_old, old_lines, new.splitlines(), fn_new, skip_header=kind == "annotation",
                    skip_blank=relocated_cells,
                )
                if out.query_spans.get(path, ()) is not None:
                    safe = body_spans(fn_old, fn_new, spans)
                    if safe:
                        out.query_spans.setdefault(path, []).extend(safe)
                    else:
                        out.query_spans[path] = None
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
                all_body_only &= bool(kind)
                # Def header changed or function deleted: all tests executing any line of function body
                out.direct |= hits.get(q, 0)
                if not revised and method not in plain_methods:
                    l_hits = self.line_hits(path)
                    for ln in range(old[3], old[4] + 1):
                        out.direct |= l_hits.get(ln, 0)
            else:
                all_body_only = False
                same = self.index["by_name"].get(method.split("#")[0], 0)
                out.broad |= (same & runtime) or same
            if old and revised and old[1] != revised[1]:
                if not kind:
                    out.query_spans[path] = None
                    dirty.add(owner)
                elif out.query_spans.get(path, ()) is not None and kind != "annotation":
                    before_node = _get_func_node(old_tree, q)
                    out.query_spans.setdefault(path, []).append((before_node.body[0].lineno, old[4]))
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
        if out.query_spans.get(path) and not same_layout():
            out.query_spans[path] = None
        literal_default_only = (query_defaults and old_scopes == new_scopes
                                and old_funcs.keys() == new_funcs.keys()
                                and all(q in query_defaults or old[2] == new_funcs[q][2]
                                        for q, old in old_funcs.items())
                                and same_layout())
        if literal_default_only:
            # Preserve the ordinary import candidates. Only tests with
            # complete case coverage can separate a def from its callers.
            out.query_spans[path] = list(query_defaults.values())
            out.query_heavy_only = True
            out.query_blocked |= self.annotation_observers()
            for name in ("__defaults__", "__kwdefaults__", "gi_frame", "cr_frame", "ag_frame"):
                out.query_blocked |= self.state_refs.mask(name)
        if prioritize and (changed or dirty):
            out.priority |= self.changed_line_hits(path, new)
        def compute_broad(limit_mask, *, include_probe=False):
            broad = 0
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
                touched = _changed_old_lines(base_bytes, new)
                if configuration:
                    # Ignore declaration lines accounted for by their readers.
                    # This includes unchanged neighbors of inserted flags.
                    body_lines = set()
                    for q, (_, _, signature, start, end) in old_funcs.items():
                        if signature != new_funcs[q][2]:
                            body_lines.update(range(start, end + 1))
                    touched &= body_lines
                elif plain_methods:
                    # A new def between two methods did not change either
                    # neighbor's body. Attribute only actual body edits.
                    touched = set()
                    before_lines = base_bytes.splitlines(keepends=True)
                    after_lines = new.splitlines(keepends=True)
                    for q, (_, _, signature, start, end) in old_funcs.items():
                        revised = new_funcs.get(q)
                        if revised is None:
                            node = _get_func_node(self.old_tree_and_lines(path)[0], q)
                            touched.update(range(node.body[0].lineno, end + 1))
                        elif signature != revised[2]:
                            left = b"".join(before_lines[start - 1:end])
                            right = b"".join(after_lines[revised[3] - 1:revised[4]])
                            touched.update(start + n - 1 for n in _changed_old_lines(left, right) if 0 < n <= end - start + 1)
                for ln in touched:
                    if ln in skip or ln in equivalent:
                        continue
                    if (include_probe and not scope_edit and changed and all_body_only and single_edit and probe
                            and (imp_hits.get(ln, 0) & limit_mask) and ln in imp_mods
                            and all(m.startswith("sqlglot/") for m in imp_mods[ln])):
                        if import_clean is None:
                            status, rows = probe()
                            if status == "broken":
                                out.mark_broken("body-state-broken", path)
                                out.broad |= coverage | self.importers[path]
                                return None
                            import_clean = (status == "ok" and all((r[0], r[1]) in changed_symbols for r in rows))
                        if import_clean:
                            broad |= c_hits.get(ln, 0) & limit_mask
                            continue
                    broad |= import_rows.get(ln, 0) & limit_mask
            if not dirty:
                return broad
            # A heap observer is a candidate if it executes the changed code (normal coverage rule),
            # or the change is module-level/class-level in a module it imports, or the module is broken.
            heap = self.index["heap_observers"] & self.importers[path]
            if not (path.startswith(("sqlglot/dialects/", "sqlglot/generators/")) or path == "sqlglot/generator.py"):
                # Heap observers only see allocations made while they run. Data edits elsewhere reach them
                # only through code they execute (runtime coverage of this module). test_generator_memory
                # builds every dialect's generator, so dialect/generator modules stay included.
                heap &= runtime
            broad |= heap
            if test_file:
                for owner in dirty:
                    affected = self.class_tests[path + "::" + owner] if owner else 0
                    broad |= affected or (self.file_tests[path] | coverage | self.importers[path])
                return broad
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
                    broad |= cls_mask & limit_mask
                    return broad
            dialect = Path(path).stem
            broad |= runtime & limit_mask
            if path.startswith("sqlglot/") and dialect in self.dialects:
                broad |= (self.dialects[dialect] | self.index["lazy_imports"].get(path, 0)) & limit_mask
            else:
                # Module-level non-class data changes keep the conservative rule.
                broad |= (coverage | self.importers[path]) & limit_mask
            return broad

        def skip_reason():
            # Rule 1 (empty): single-file edit with no direct/broad/coverage/importers/static-broad tests
            # and no dict-read fallback tests (incomplete==0 and no tracked dict reads on this path).
            # Every probe outcome ("ok" with any rows, "broken", "unknown") produces the empty set.
            if single_edit and not out.mask and not (coverage | self.importers[path]) and compute_broad(-1) == 0:
                if changed or not self.dict_reads or (
                        not self.dict_reads["incomplete"]
                        and all(not e["whole"] and not e["keys"] and not e["import_whole"] and not e["import_keys"]
                                for (p, _, _), e in self.dict_reads["dicts"].items() if p == path)):
                    return "empty", True

            static_broad = compute_broad(-1)
            broken_included = ((coverage | self.importers[path]) & ~(out.mask | static_broad)) == 0

            # Rule 2 (changed_full): when a function also changed (changed=True), Site 1 skips
            # "if status == 'ok' and not changed", so "ok" (any rows) and "unknown" both fall through
            # to compute_broad(-1). When single_edit holds, coverage|importers[path] is already included,
            # the schedule is already at full-catalog price, and entry_edits cannot witness a top-level
            # None->bool default edit, setting broken=True matches "broken" with zero schedule penalty.
            if changed:
                if (single_edit and not literal_default_only and broken_included
                        and self._is_full_schedule(out.mask | static_broad)):
                    from .patch_proof import entry_edits
                    old_tree, _ = self.old_tree_and_lines(path)
                    if not entry_edits(old_tree, ast.parse(new, path)):
                        return "changed_full", True
                return None, False

            old_tree, _ = self.old_tree_and_lines(path)
            touched = _changed_old_lines(base_src, new)

            # Rule 3 (docstring): single-file edit changing only a non-empty Module/ClassDef docstring
            # literal (bytes outside the literal identical). Under "ok" (any rows) and "unknown", all
            # post-probe refinements fail so selection falls through to compute_broad(-1). Requiring
            # broken_included and full-catalog schedule with broken=True makes the skipped selection
            # a strict superset even of "broken" with zero schedule penalty (and in pinned sqlglot,
            # only Expr.__init_subclass__ touches __doc__, preserving truthiness for non-empty strings).
            if (single_edit and broken_included and self._is_full_schedule(out.mask | static_broad)
                    and _is_docstring_only(old_tree, new_tree_for_stmts, base_src, new, touched)):
                return "docstring", True

            # Rule 4 (arg_types): single-file edit on tracked Expression subclasses (is_expr_cls,
            # "" not in dirty) replacing only one arg_types literal with another dict[str, bool].
            # - "ok" (any rows): is_expr_cls bypasses _dict_value_candidates; data_only is True and
            #   state_refs.readers(rows) returns None on lowercase arg_types/required_args (or <= -1),
            #   falling through to compute_broad(-1) (or 0 <= compute_broad(-1) if rows=[]).
            # - "unknown": falls through to compute_broad(-1).
            # - "broken": impossible in pinned sqlglot because every import-time consumer of arg_types
            #   (Expr.__init_subclass__, **Cast.arg_types, set(exp.Select.arg_types)) accepts any
            #   dict[str, bool] without raising.
            is_expr_cls = (
                path.startswith("sqlglot/expressions/")
                and "" not in dirty
                and self.expr_owners(path, dirty)
                and all(f"{path}::{o}" in self.index.get("class_usage", {}) for o in dirty)
            )
            if single_edit and is_expr_cls and data_only(old_tree, new_tree_for_stmts, base_src, new, touched):
                targets, ov, nv = _changed_assign(old_tree, new_tree_for_stmts, base_src, new, touched)
                if (targets == ["arg_types"]
                        and _is_const_bool_dict(ov) and _is_const_bool_dict(nv)
                        and list(ast.literal_eval(ov).items()) != list(ast.literal_eval(nv).items())):
                    return "arg_types", False

            return None, False

        refined = None
        if dirty and scope_edit and probe:
            reason, mark_broken = skip_reason()
            if reason is not None:
                if mark_broken:
                    out.mark_broken(f"state-skipped-{reason}", path)
                print(f"state {path} status=skipped reason={reason}", file=sys.stderr)
            else:
                status, rows = probe()
                if status == "broken":
                    out.mark_broken("scope-state-broken", path)
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
        broad = compute_broad(limit, include_probe=True)
        if broad is None:
            return
        out.broad |= broad


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


def _inert_dict_value(node):
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Tuple):
        return all(_inert_dict_value(item) for item in node.elts)
    if isinstance(node, ast.Lambda):
        return all(value is None or isinstance(value, ast.Constant)
                   for value in (*node.args.defaults, *node.args.kw_defaults))
    return False


def _literal_string_keys(mapping):
    explicit = [key for key in mapping.keys if key is not None]
    if not all(isinstance(key, ast.Constant) and type(key.value) is str for key in explicit):
        return None
    keys = {key.value for key in explicit}
    return keys if len(keys) == len(explicit) else None


def _restore_dict_order(old, revised, keys):
    """Restore a permutation of inert entries before any unchanged unpacking."""
    size = len(keys)
    old_order = [node.value if node is not None else None for node in old.keys[:size]]
    new_order = [node.value if node is not None else None for node in revised.keys[:size]]
    if old_order == new_order or None in old_order or None in new_order:
        return False
    for i, (before, after) in enumerate(zip(old_order, new_order)):
        if before != after and not (_inert_dict_value(old.values[i]) and _inert_dict_value(revised.values[i])):
            return False
    # Comparing the complete AST after restoration checks values, unpacking
    # and every other statement. Consumers of iteration order remain selected.
    by_key = {key: (revised.keys[i], revised.values[i]) for i, key in enumerate(new_order)}
    revised.keys[:size] = [by_key[key][0] for key in old_order]
    revised.values[:size] = [by_key[key][1] for key in old_order]
    return True


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
