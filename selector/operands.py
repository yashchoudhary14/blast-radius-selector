"""Intersect one changed function's coverage with its pristine operand evidence."""
import ast
import marshal
from functools import lru_cache
from zipfile import ZipFile

OPS = ("Eq", "NotEq", "Lt", "LtE", "Gt", "GtE", "Is", "IsNot", "In", "NotIn")


def edits(old, new):
    found = []

    def match(a, b):
        if type(a) is not type(b):
            return False
        if isinstance(a, list):
            return len(a) == len(b) and all(match(x, y) for x, y in zip(a, b))
        if not isinstance(a, ast.AST):
            return a == b
        if isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return ast.dump(a) == ast.dump(b)
        if isinstance(a, ast.Constant):
            # AST fields such as 0, False and 0.0 compare equal as Python
            # objects even though substituting one literal changes code.
            return ast.dump(a) == ast.dump(b)
        if isinstance(a, ast.Compare) and len(a.ops) == len(b.ops) == 1:
            prior, after = (OPS.index(type(n.ops[0]).__name__) for n in (a, b))
            left, right = a.left, a.comparators[0]
            if prior != after:
                found.append((a, ("operator", prior, after)))
                return match(left, b.left) and match(right, b.comparators[0])
            for pos, (x, y) in enumerate(((left, b.left), (right, b.comparators[0]))):
                if (isinstance(x, ast.Constant) and isinstance(y, ast.Constant)
                        and type(x.value) is type(y.value) and type(x.value) in (str, int, float)
                        and x.value != y.value and prior < 6):
                    # Only the nonliteral operand was summarized.
                    if isinstance(right if pos == 0 else left, ast.Constant):
                        return False
                    found.append((a, ("constant", prior, pos, x.value, y.value)))
                    return (match(right, b.comparators[0]) if pos == 0 else match(left, b.left))
        if isinstance(a, ast.BoolOp) and type(a.op) is not type(b.op):
            found.append((a, ("boolean",)))
            return match(a.values, b.values)
        return all(match(getattr(a, field), getattr(b, field)) for field in a._fields)

    return found if old and new and match(old.body, new.body) else []


def excludes(bounds, op, pos, old, new):
    if bounds is None or type(bounds[0]) is not type(old):
        return False
    low, high = bounds
    a, b = sorted((old, new))
    if op <= 1:
        return not (low <= a <= high or low <= b <= high)
    if pos == 0:
        op = {2: 4, 3: 5, 4: 2, 5: 3}[op]
    # x < k and x >= k change on [a,b). The other two change on (a,b].
    return high < a or low >= b if op in (2, 5) else high <= a or low > b


def unchanged(rule, vectors, bounds):
    if 0 in vectors:
        return False
    if rule[0] == "boolean":
        return vectors == (1,)
    if rule[0] == "constant":
        return excludes(bounds, *rule[1:])
    _, old, new = rule
    required = (1 << old) | (1 << new)
    return all((v >> 10) & required == required and (v & required) in (0, required) for v in vectors)


@lru_cache(maxsize=4)
def archive(file):
    return ZipFile(file)


@lru_cache(maxsize=16)
def member(file, name):
    return marshal.loads(archive(file).read(name))


def narrow(maps, path, old, new, mask, line_hits):
    rules = edits(old, new)
    if not rules:
        return mask
    file = maps / "operands.zip"
    if not file.is_file() or path not in archive(file).NameToInfo:
        return mask
    rows = member(file, path)
    affected = mask & ~member(file, "complete")
    for site, rule in rules:
        key = site.lineno, site.col_offset, type(site).__name__, site.end_col_offset
        if key not in rows:
            return mask
        observed = 0
        for vectors, bounds, tests in rows[key]:
            reached = tests & mask
            observed |= reached
            if reached and not unchanged(rule, vectors, bounds):
                affected |= reached
        # Coverage and operand collection use different processes. Retain
        # any candidate that reached this line only in the coverage run.
        for line in range(site.lineno, site.end_lineno + 1):
            affected |= line_hits.get(line, 0) & mask & ~observed
    return affected
