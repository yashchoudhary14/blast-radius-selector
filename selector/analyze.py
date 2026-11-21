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
from .scopes import scope_changed, snapshot

COMPOUND = (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith, ast.Try, ast.Match)
try:
    COMPOUND = (*COMPOUND, ast.TryStar)
except AttributeError:
    pass


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
            spans.append(h_sp)
            before_stmt = after_stmt = None
            for s in body:
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
    python_paths: list[str] = field(default_factory=list)
    dialect_subset_safe: bool = True

    @property
    def mask(self):
        return self.direct | self.broad


def numbers(mask):
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


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
        with gzip.open(maps / "import_lines.json.gz") as source:
            imports = json.load(source)
        masks = []
        for modules in imports["sets"]:
            mask = 0
            for module in modules:
                mask |= self.importers[module]
            masks.append(mask)
        self.indirect_hits = {path: {int(ln): masks[idx] for ln, idx in rows.items()}
                              for path, rows in imports["files"].items()}
        # Lines executed only while their own module is imported (class/module bodies defining it).
        self.import_self_only = {path: {int(ln) for ln, idx in rows.items() if imports["sets"][idx] == [path]}
                                 for path, rows in imports["files"].items()}
        for cache in imports["caches"]:
            users = 0
            for path, ln in cache["reads"]:
                users |= self.line_hits(path).get(ln, 0) | self.indirect_hits.get(path, {}).get(ln, 0)
            for path, lines in cache["files"].items():
                rows = self.indirect_hits.setdefault(path, {})
                for ln in lines:
                    rows[ln] = rows.get(ln, 0) | users
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

    def analyze(self, patch, *, prioritize=False):
        out = Candidates()
        for path, new in changes(self.base, patch).items():
            p = Path(path)
            readers = self.index["reads"].get(path, 0)
            out.direct |= readers
            out.priority |= readers
            if new is None or not (self.base / path).exists():
                out.dialect_subset_safe = False
                directory = p.parent
                while not (self.base / directory).is_dir():
                    directory = directory.parent
                out.broad |= self.index["dirs"].get(directory.as_posix(), 0)
            if p.suffix == ".py":
                out.python_paths.append(path)
                if not path.startswith("sqlglot/"):
                    out.dialect_subset_safe = False
                self.python(path, new, out, prioritize=prioritize)
        return out

    def python(self, path, new, out, *, prioritize=False):
        coverage = self.index["all_hits"].get(path, 0)
        try:
            if new is None:
                raise SyntaxError("deleted module")
            # ast.parse alone accepts constructs such as return outside a function.
            compile(new, path, "exec", dont_inherit=True)
            tree = ast.parse(new, path)
            new_tree_for_stmts = ast.parse(new, path)
        except (ValueError, SyntaxError):
            out.broad |= coverage | self.importers[path] | self.file_tests[path]
            out.broken = True
            return
        old_funcs, old_scopes = self.old(path)
        new_funcs, new_scopes = snapshot(tree, new, old_funcs)
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
        if dirty:
            # Class/module-level edits that sit entirely inside lambda bodies (e.g. entries of
            # Parser.FUNCTIONS / Generator.TRANSFORMS dicts) are code, not data: a lambda body runs
            # at call time, so per-test line coverage of those lines names every caller.
            base_src = (self.base / path).read_bytes() if (self.base / path).is_file() else b""
            _, old_masked = snapshot(_MaskLambdas().visit(ast.parse(base_src, path)), base_src)
            _, new_masked = snapshot(_MaskLambdas().visit(ast.parse(new, path)), new)
            lambda_only = {q for q in dirty if q in old_masked and q in new_masked
                           and not scope_changed(old_masked[q], new_masked[q])}
            if lambda_only:
                dirty -= lambda_only
                l_hits = self.line_hits(path)
                for ln in _changed_old_lines(base_src, new):
                    out.direct |= l_hits.get(ln, 0)
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
            if old and revised and old[1] == revised[1]:
                # Only function body changed: use statement-level granularity
                old_tree, old_lines = self.old_tree_and_lines(path)
                fn_old = _get_func_node(old_tree, q)
                fn_new = _get_func_node(new_tree_for_stmts, q)
                spans = _changed_stmt_spans(fn_old, old_lines, new.splitlines(), fn_new)
                l_hits = self.line_hits(path)
                if spans and l_hits:
                    stmt_mask = 0
                    for s_start, s_end in spans:
                        for ln in range(s_start, s_end + 1):
                            stmt_mask |= l_hits.get(ln, 0)
                    out.direct |= stmt_mask
                else:
                    out.direct |= hits.get(q, 0)
            elif old:
                # Def header changed or function deleted: all tests executing any line of function body
                out.direct |= hits.get(q, 0)
                if not revised:
                    l_hits = self.line_hits(path)
                    for ln in range(old[3], old[4] + 1):
                        out.direct |= l_hits.get(ln, 0)
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
        if prioritize and (changed or dirty):
            out.priority |= self.changed_line_hits(path, new)
        import_rows = self.indirect_hits.get(path)
        if import_rows and (scope_edit or changed):
            base_bytes = (self.base / path).read_bytes()
            skip = set()
            usage = self.index.get("class_usage", {})
            if (dirty and "" not in dirty and usage and path.startswith("sqlglot/expressions/")
                    and all(f"{path}::{o}" in usage for o in dirty) and self.expr_owners(path, dirty)):
                # Expression class-body data (arg_types, flags) executes only while its own module is
                # imported; its effect reaches tests through the class, which the class-usage refinement
                # below selects. Function bodies run at import time keep the importer attribution.
                skip = self.import_self_only.get(path, set()) - _function_lines(base_bytes, path)
            for ln in _changed_old_lines(base_bytes, new):
                if ln not in skip:
                    out.broad |= import_rows.get(ln, 0)
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
                out.broad |= cls_mask
                return
        dialect = Path(path).stem
        out.broad |= runtime
        if path.startswith("sqlglot/") and dialect in self.dialects:
            out.broad |= self.dialects[dialect] | self.index["lazy_imports"].get(path, 0)
        else:
            # Module-level non-class data changes keep the conservative rule.
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
    """Replace every lambda with a constant so scope signatures ignore lambda-internal edits."""

    def visit_Lambda(self, node):
        return ast.copy_location(ast.Constant(value="<lambda>"), node)


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


if __name__ == "__main__":
    raise SystemExit(main())
