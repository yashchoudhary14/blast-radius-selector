"""Use observations of the entire pristine method, including imports and spawned tasks."""
import ast
import functools
import gzip
import json
from bisect import bisect_left
from pathlib import Path

from runner.overlay import changes

from .analyze import _header_span
from .fixture_cases import FIXTURES, TPC_DS, TPC_H
from .scopes import snapshot
from .source_equivalence import _same_directives_text_only


def same(left, right):
    return ast.dump(left) == ast.dump(right)


def leading_guard(fn):
    # Inserting this statement cannot introduce a local, free variable or call.
    pos = int(ast.get_docstring(fn, clean=False) is not None)
    if pos >= len(fn.body):
        return None
    stmt = fn.body[pos]
    if not (isinstance(stmt, ast.If) and not stmt.orelse and len(stmt.body) == 1
            and isinstance(stmt.body[0], ast.Return)
            and (stmt.body[0].value is None or isinstance(stmt.body[0].value, ast.Constant))):
        return None
    expr = stmt.test
    negated = isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not)
    operand = expr.operand if negated else expr
    if isinstance(operand, ast.Name):
        return pos, operand.id, 2 if negated else 4
    return None


def entry_edits(old, new):
    """Undo supported edits, then compare the complete AST to catch any other edits."""
    def functions(body, prefix=""):
        for node in body:
            if isinstance(node, ast.FunctionDef):
                yield prefix + node.name, node
            elif isinstance(node, ast.ClassDef):
                yield from functions(node.body, prefix + node.name + ".")

    old_funcs = dict(functions(old.body))
    new_funcs = dict(functions(new.body))
    rules = []
    for name, fn in new_funcs.items():
        prior = old_funcs.get(name)
        if prior is None or prior.decorator_list or fn.decorator_list:
            continue
        guard = leading_guard(fn) if len(fn.body) == len(prior.body) + 1 else None
        defaults = fn.args.defaults
        changes = [
            i for i, (before, after) in enumerate(zip(prior.args.defaults, defaults))
            if isinstance(before, ast.Constant) and before.value is None
            and isinstance(after, ast.Constant) and type(after.value) is bool
        ] if len(defaults) == len(prior.args.defaults) else []
        if not (guard or changes) or any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(prior)):
            continue
        args = [a.arg for a in (*fn.args.posonlyargs, *fn.args.args)]
        if guard:
            pos, param, mask = guard
            if param in args or param in [a.arg for a in fn.args.kwonlyargs]:
                fn.body.pop(pos)
                rules.append((name, prior.lineno, prior.body[0].lineno, prior.body[-1].end_lineno,
                              param, mask, "guard", ()))
        for i in changes:
            defaults[i] = prior.args.defaults[i]
            param = args[i-len(defaults)]
            reads = tuple((node.lineno, node.end_lineno) for stmt in prior.body for node in ast.walk(stmt)
                          if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == param)
            rules.append((name, prior.lineno, prior.body[0].lineno, prior.body[-1].end_lineno,
                          param, 1, "default", reads))
    return rules if rules and same(old, new) else None


def identity_edits(old, new):
    """Verify the entire patch consists of parity-preserving comparison substitutions."""
    left = [n for n in ast.walk(old) if isinstance(n, ast.Compare)]
    right = [n for n in ast.walk(new) if isinstance(n, ast.Compare)]
    if len(left) != len(right):
        return None
    sites = []
    for a, b in zip(left, right):
        if len(a.ops) == len(b.ops) == 1 and (
                (isinstance(a.ops[0], ast.Is) and isinstance(b.ops[0], ast.Eq)) or
                (isinstance(a.ops[0], ast.IsNot) and isinstance(b.ops[0], ast.NotEq))):
            b.ops = a.ops
            sites.append(f"{a.lineno}:{a.col_offset}")
    return sites if sites and same(old, new) else None


@functools.lru_cache(maxsize=5)
def evidence(number):
    file = Path(__file__).with_name("witness_build") / f"calls-{number}.json.gz"
    loaded, calls, meta, lines, reads, comparisons = json.loads(gzip.decompress(file.read_bytes()))
    return set(loaded) | lines.keys(), calls, set(meta), set(reads), lines, set(comparisons)


class PatchProof:
    def __init__(self, analyzer, patch, isolated_tests=()):
        self.files = {}
        self.fixture_companions = set()
        self.classified = {}
        self.analyzer = analyzer
        edits = changes(analyzer.base, patch)
        for path, new in edits.items():
            if path in isolated_tests:
                continue
            base = analyzer.base / path
            if new is None or not base.is_file():
                self.files.clear()
                break
            if path in edits.modes and edits.modes[path] != base.stat().st_mode & 0o777:
                self.files.clear()
                break
            if path.startswith(FIXTURES) and path.endswith(".sql") and path not in {TPC_DS, TPC_H}:
                self.fixture_companions.add(path)
                continue
            if not (path.startswith("sqlglot/") and path.endswith(".py") and new):
                self.files.clear()
                break
            self.files[path] = (base.read_bytes(), new)

    def classify(self, path, rule):
        key = (path, rule)
        if key not in self.classified:
            original, _ = self.analyzer.old_tree_and_lines(path)
            self.classified[key] = rule(original, ast.parse(self.files[path][1]))
        return self.classified[key]

    def witness(self, number):
        bit = 1 << number
        if not self.files or any(self.analyzer.index["reads"].get(path, 0) & bit for path in self.fixture_companions):
            return None
        loaded, calls, meta, reads, covered, comparisons = evidence(number)
        reasons = set()
        for path in self.files:
            if path in reads or self.analyzer.index["reads"].get(path, 0) & bit:
                return None
            if path not in loaded:
                return None
            sites = self.classify(path, identity_edits)
            if sites and all(f"{path}:{site}" in comparisons for site in sites):
                reasons.add("identity")
                continue
            rules = self.classify(path, entry_edits)
            if not rules:
                return None
            if any("." in row[0] for row in rules) and not _same_directives_text_only(*self.files[path]):
                return None
            module = path[:-3].replace("/", ".").removesuffix(".__init__")
            for name, line, body_start, end, param, mask, mode, read_spans in rules:
                symbol = f"{module}.{name}"
                if symbol + ".metadata" in meta or (mode == "default" and symbol + ".signature" in meta):
                    return None
                entry = calls.get(f"{path}:{line}")
                if entry is None and not any(body_start <= loc <= end for loc in covered.get(path, ())):
                    reasons.add("uncalled")
                    continue
                if entry and entry.get(param, 0) & mask:
                    reasons.add(mode)
                    continue
                lines = covered.get(path, ())
                if (mode == "default" and not any(
                        self.analyzer.state_refs.mask(observer) & bit for observer in (
                            "locals", "vars", "eval", "exec", "f_locals", "currentframe", "_getframe",
                            "getouterframes", "gi_frame", "cr_frame", "ag_frame"))
                        and not any((i := bisect_left(lines, a)) < len(lines) and lines[i] <= b
                                    for a, b in read_spans)):
                    reasons.add("default-unread")
                    continue
                return None
        return "+".join(sorted(reasons))

    @functools.cached_property
    def spans(self):
        # Guard.add independently checks every AST change in the subprocess.
        # At ambiguous insertions or signature/data edits, retain the candidate.
        from difflib import SequenceMatcher

        if self.fixture_companions:
            return None, "fixture-companion"
        spans = {}
        for path, (old, new) in self.files.items():
            old_tree, before = self.analyzer.old_tree_and_lines(path)
            funcs, scopes = self.analyzer.old(path)
            revised, new_scopes = snapshot(ast.parse(new), new)
            if scopes != new_scopes or funcs.keys() != revised.keys():
                return None, "data"
            intervals = []
            for name, (_, header, body, start, end) in funcs.items():
                if revised[name][1] != header:
                    return None, "signature"
                if revised[name][2] != body:
                    intervals.append((start, end))
            after = new.splitlines()
            if len(before) == len(after):
                changed = [i for i, (a, b) in enumerate(zip(before, after), 1) if a != b]
            else:
                changed = []
                for op, i, j, k, l in SequenceMatcher(None, before, after).get_opcodes():
                    if op == "equal":
                        continue
                    if op != "replace":
                        return None, "insertion-or-deletion"
                    changed.extend(range(i+1, j+1))
            if not changed or not all(any(a <= i <= b for a,b in intervals) for i in changed):
                return None, "outside-body"
            statements = [_header_span(n) for n in ast.walk(old_tree) if isinstance(n, ast.stmt)]
            owners = set()
            for line in changed:
                enclosing = [(a,b) for a,b in statements if a <= line <= b]
                if not enclosing:
                    return None, "ambiguous-statement"
                width = min(b-a for a,b in enclosing)
                owners.update((a,b) for a,b in enclosing if b-a == width)
            spans[path] = sorted(owners)
        return (spans, "replay") if spans else (None, "non-package")
