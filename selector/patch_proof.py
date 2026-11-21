"""Use observations of the entire pristine method, including imports and spawned tasks."""
import ast
import functools
import gzip
import json
from pathlib import Path

from runner.overlay import changes

from .analyze import _header_span
from .scopes import snapshot


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
    old_funcs = {f.name: f for f in old.body if isinstance(f, ast.FunctionDef)}
    new_funcs = {f.name: f for f in new.body if isinstance(f, ast.FunctionDef)}
    rules = []
    for name, fn in new_funcs.items():
        prior = old_funcs.get(name)
        if prior is None or prior.decorator_list or fn.decorator_list or any(
                isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(prior)):
            continue
        args = [a.arg for a in (*fn.args.posonlyargs, *fn.args.args)]
        if len(fn.body) == len(prior.body) + 1:
            guard = leading_guard(fn)
            if guard:
                pos, param, mask = guard
                if param in args or param in [a.arg for a in fn.args.kwonlyargs]:
                    fn.body.pop(pos)
                    rules.append((name, prior.lineno, prior.body[0].lineno, prior.body[-1].end_lineno, param, mask, "guard"))
        defaults = fn.args.defaults
        if defaults and len(defaults) == len(prior.args.defaults):
            for i, before in enumerate(prior.args.defaults):
                after = defaults[i]
                if (isinstance(before, ast.Constant) and before.value is None
                        and isinstance(after, ast.Constant) and type(after.value) is bool):
                    defaults[i] = before
                    rules.append((name, prior.lineno, prior.body[0].lineno, prior.body[-1].end_lineno, args[i-len(defaults)], 1, "default"))
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
        self.analyzer = analyzer
        for path, new in changes(analyzer.base, patch).items():
            if path in isolated_tests:
                continue
            base = analyzer.base / path
            if not (path.startswith("sqlglot/") and path.endswith(".py") and new and base.is_file()):
                self.files.clear()
                break
            self.files[path] = (base.read_bytes(), new)

    @functools.cached_property
    def rules(self):
        return {p: entry_edits(self.analyzer.old_tree_and_lines(p)[0], ast.parse(new))
                for p, (_, new) in self.files.items()}

    @functools.cached_property
    def identity(self):
        return {p: identity_edits(self.analyzer.old_tree_and_lines(p)[0], ast.parse(new))
                for p, (_, new) in self.files.items()}

    def witness(self, number):
        if not self.files:
            return None
        loaded, calls, meta, reads, covered, comparisons = evidence(number)
        reasons = set()
        for path in self.files:
            if path in reads or self.analyzer.index["reads"].get(path, 0) & (1 << number):
                return None
            if path not in loaded:
                return None
            sites = self.identity[path]
            if sites and all(f"{path}:{site}" in comparisons for site in sites):
                reasons.add("identity")
                continue
            rules = self.rules[path]
            if not rules:
                return None
            module = path[:-3].replace("/", ".").removesuffix(".__init__")
            for name, line, body_start, end, param, mask, mode in rules:
                symbol = f"{module}.{name}"
                if symbol + ".metadata" in meta or (mode == "default" and symbol + ".signature" in meta):
                    return None
                entry = calls.get(f"{path}:{line}")
                if entry is None and not any(body_start <= loc <= end for loc in covered.get(path, ())):
                    reasons.add("uncalled")
                    continue
                if not entry or not entry.get(param, 0) & mask:
                    return None
                reasons.add(mode)
        return "+".join(sorted(reasons))

    @functools.cached_property
    def spans(self):
        # Guard.add independently checks every AST change in the subprocess.
        # At ambiguous insertions or signature/data edits, retain the candidate.
        from difflib import SequenceMatcher

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
