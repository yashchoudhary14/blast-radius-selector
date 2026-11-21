"""Pinned source references, including literal reflection and unresolved dynamic reads."""
import ast
import dis
import marshal
import re
import sys
import types
from collections import defaultdict
from functools import lru_cache

from runner.state import code_parts
from .collect import BASE, MAPS
from .scopes import digest


def signature(code):
    def const(value):
        if isinstance(value, types.CodeType):
            return ("code", signature(value))
        if isinstance(value, frozenset):
            return ("set", sorted(map(const, value), key=repr))
        return value

    parts = code_parts(code, const)
    if (code.co_flags & 2 and code.co_consts and isinstance(code.co_consts[0], str)
            and not any(i.opcode in dis.hasconst and i.arg == 0 for i in dis.get_instructions(code))):
        # A literal used by the body can share the docstring's constant slot.
        parts = (parts[0], (None, *parts[1][1:]), *parts[2:])
    return digest(marshal.dumps(parts, 2))


def functions(code):
    found = {}
    def visit(parent):
        for nested in parent.co_consts:
            if isinstance(nested, types.CodeType):
                key = nested.co_qualname.replace(".<locals>.", ".")
                found[key] = None if key in found else nested
                visit(nested)
    visit(code)
    return found


class References(ast.NodeVisitor):
    def __init__(self, path, tree):
        self.lines = defaultdict(set)
        self.fields = set()
        self.scope = ""
        self.path = path
        self.parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

    def dynamic(self, node):
        """Bound reflection by its receiver or by the names accepted in pinned dispatch loops."""
        path, scope = self.path, self.scope
        if (path, scope) in {("sqlglot/generator.py", "_build_dispatch"),
                             ("sqlglot/optimizer/simplify.py", "_build_gen_dispatch")}:
            # Both loops admit only names ending in _sql.
            return
        if (path, scope) in {("sqlglot/helper.py", "subclasses"),
                             ("sqlglot/dialects/dialect.py", "_try_load")}:
            # These searches accept only class objects. readers() admits data bindings.
            return
        domain = "*"
        if scope == "__getattr__" or path == "sqlglot/serde.py":
            domain = "*module"
        elif path == "sqlglot/tokenizer_core.py" and scope == "__repr__":
            domain = "*token"
        elif path == "sqlglot/executor/env.py":
            # EXTRACT reflects on the SQL value furnished to the executor.
            domain = "*sqlvalue"
        self.lines[domain].add(node.lineno)

    def visit_FunctionDef(self, node):
        old, self.scope = self.scope, node.name
        self.generic_visit(node)
        self.scope = old

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Attribute(self, node):
        if isinstance(node.ctx, ast.Load):
            self.fields.add(node.attr)
            self.lines[node.attr].update((node.lineno, node.end_lineno))
            if node.attr == "__dict__":
                parent = self.parents[node]
                literal = None
                if isinstance(parent, ast.Attribute) and parent.attr in ("get", "pop"):
                    call = self.parents[parent]
                    if isinstance(call, ast.Call) and call.args:
                        literal = call.args[0]
                elif isinstance(parent, ast.Compare) and parent.comparators == [node]:
                    literal = parent.left
                if isinstance(literal, ast.Constant) and isinstance(literal.value, str):
                    self.lines[literal.value].add(node.lineno)
                else:
                    self.dynamic(node)
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.lines[node.id].add(node.lineno)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            # Star imports copy module bindings, including aliases, into the captured graph.
            if alias.name != "*":
                self.lines[alias.name].add(node.lineno)

    def visit_Call(self, node):
        fn = node.func
        callee = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
        if callee in ("getattr", "hasattr", "setattr", "delattr") and len(node.args) >= 2:
            attr = node.args[1]
            if isinstance(attr, ast.Constant) and isinstance(attr.value, str):
                self.fields.add(attr.value)
                self.lines[attr.value].add(node.lineno)
            elif ((isinstance(attr, ast.JoinedStr) and isinstance(attr.values[-1], ast.Constant)
                   and str(attr.values[-1].value).endswith("_sql"))
                  or (isinstance(attr, ast.BinOp) and isinstance(attr.op, ast.Add)
                      and isinstance(attr.right, ast.Constant) and attr.right.value == "_sql")):
                pass  # Method dispatch; these names cannot match data eligible for refinement.
            else:
                self.dynamic(node)
        elif callee in ("vars", "dir", "getmembers", "__subclasses__", "globals", "locals"):
            self.dynamic(node)
        self.generic_visit(node)


def build():
    refs = defaultdict(dict)
    fields = set()
    for folder in ("sqlglot", "tests"):
        for file in sorted((BASE / folder).rglob("*.py")):
            path = file.relative_to(BASE).as_posix()
            source = file.read_bytes()
            tree = ast.parse(source, path)
            visitor = References(path, tree)
            visitor.visit(tree)
            fields.update(visitor.fields)
            for name, lines in visitor.lines.items():
                refs[name][path] = tuple(sorted(lines))
            compiled = compile(source, path, "exec", dont_inherit=True)
            cache = MAPS / "ast" / (path + ".code")
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(marshal.dumps({q: signature(c) for q, c in functions(compiled).items() if c}))
    (MAPS / "name-refs.bin").write_bytes(marshal.dumps((dict(refs), fields)))


class StateRefs:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.refs, self.fields = marshal.loads((analyzer.maps / "name-refs.bin").read_bytes())
        self.keys = marshal.loads((analyzer.maps / "key-reads.bin").read_bytes())

    @lru_cache(maxsize=512)
    def mask(self, name, keys=None):
        mask = 0
        for path, lines in self.refs.get(name, {}).items():
            direct = self.analyzer.line_hits(path)
            imports = self.analyzer.indirect_hits.get(path, {})
            for line in lines:
                observed = self.keys.get((name, path, line)) if keys else None
                hits = direct.get(line, 0)
                if observed:
                    hits &= ~observed[None]
                    for key in keys:
                        hits |= observed.get(key, 0)
                mask |= hits | imports.get(line, 0)
        return mask

    def readers(self, rows):
        mask = 0
        for owner, attr, entries, kind, keyed in rows:
            # Enum lookup, descriptor protocols, and dynamically discovered
            # methods can reach consumers without an explicit field reference.
            if (kind == "other" or not attr.isupper() or attr.startswith("__")
                    or not self.refs.get(attr) or kind == "class" and attr not in self.fields):
                return None
            mask |= self.mask(attr, tuple(entries) if keyed else None) | self.mask("*")
            if kind == "module":
                mask |= self.mask("*module")
            if owner.startswith("sqlglot.tokenizer_core.Token"):
                mask |= self.mask("*token")
            if owner.startswith(("sqlglot.expressions.", "sqlglot.executor.")):
                mask |= self.mask("*sqlvalue")
        return mask


def _re_is_clean(tree):
    if not any(
        isinstance(s, ast.Import) and any(a.name == "re" and a.asname is None for a in s.names)
        for s in tree.body
    ):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "re" and not isinstance(node.ctx, ast.Load):
            return False
        if isinstance(node, ast.arg) and node.arg == "re":
            return False
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == "re":
            return False
    return True


def _re_flags(node):
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == "re" and node.attr.isupper()):
        val = getattr(re, node.attr, None)
        if isinstance(val, (int, re.RegexFlag)):
            return int(val)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = _re_flags(node.left)
        right = _re_flags(node.right)
        if left is not None and right is not None:
            return left | right
    return None


def _pure_re_compile(node):
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "re"
            and node.func.attr == "compile" and not node.keywords and len(node.args) in (1, 2)):
        return False
    pat = node.args[0]
    if not (isinstance(pat, ast.Constant) and type(pat.value) in (str, bytes)):
        return False
    flags = 0
    if len(node.args) == 2:
        flags = _re_flags(node.args[1])
        if flags is None:
            return False
    try:
        re.compile(pat.value, flags)
    except re.error:
        return False
    return True


def data_only(old, new, original, updated, touched, *, condition=False):
    """Prove all changed bytes belong to one data expression (or one unreached guard)."""
    def values(tree):
        for stmt in ast.walk(tree):
            if condition and isinstance(stmt, ast.If):
                yield stmt.test
            elif not condition and isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                targets = stmt.targets if isinstance(stmt, ast.Assign) else (stmt.target,)
                if stmt.value and all(isinstance(t, ast.Name) for t in targets):
                    yield stmt.value

    def outside(source, node):
        lines = source.splitlines(keepends=True)
        return (lines[:node.lineno-1] + [lines[node.lineno-1][:node.col_offset]],
                [lines[node.end_lineno-1][node.end_col_offset:]] + lines[node.end_lineno:])

    re_safe = _re_is_clean(old)
    for a, b in zip(values(old), values(new)):
        if not all(a.lineno <= n <= a.end_lineno for n in touched):
            continue
        if outside(original, a) != outside(updated, b):
            continue
        if not condition and not _data_expression(a, b, re_safe=re_safe):
            continue
        return True
    return False


def _data_expression(left, right, *, re_safe=False):
    # Accept plain literals, names and container displays. Existing calls or
    # deferred lambda bodies must be identical; they cannot introduce an
    # extra side effect solely because a neighbouring dictionary key changed.
    def opaque(node):
        items = []
        for n in ast.walk(node):
            if not isinstance(
                n, (ast.Call, ast.NamedExpr, ast.Lambda, ast.ListComp, ast.DictComp,
                    ast.SetComp, ast.GeneratorExp, ast.BinOp, ast.Subscript)
            ):
                continue
            if (isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitOr)
                    and isinstance(n.left, ast.Dict) and isinstance(n.right, ast.Dict)):
                continue
            if re_safe and _pure_re_compile(n):
                continue
            items.append(ast.dump(n))
        return sorted(items)
    return opaque(left) == opaque(right)


def same_runtime_declarations(original, updated):
    """Ignore type-only declarations, identity decorators and stdlib import permutations."""
    def normalized(source):
        tree = ast.parse(source)
        typing = {a.asname or a.name for n in tree.body if isinstance(n, ast.Import)
                  for a in n.names if a.name == "typing"}
        markers = {a.asname or a.name for n in tree.body if isinstance(n, ast.ImportFrom)
                   and n.module in ("mypy_extensions", "sqlglot.helper")
                   for a in n.names if a.name == "mypyc_attr"}
        for n in ast.walk(tree):
            if (isinstance(n, ast.If) and isinstance(n.test, ast.Attribute)
                    and n.test.attr == "TYPE_CHECKING" and isinstance(n.test.value, ast.Name)
                    and n.test.value.id in typing):
                n.body = [ast.Pass()]
            if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                n.decorator_list = [d for d in n.decorator_list if not (
                    isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id in markers
                    and all(isinstance(v, ast.Constant) for v in [*d.args, *(k.value for k in d.keywords)]))]
        # Only contiguous stdlib imports with distinct bound names may commute.
        imports, names, result = [], set(), []
        for stmt in tree.body:
            modules = ([a.name for a in stmt.names] if isinstance(stmt, ast.Import)
                       else [stmt.module or ""] if isinstance(stmt, ast.ImportFrom) and not stmt.level else [])
            if modules and all(m.partition(".")[0] in sys.stdlib_module_names and m != "__future__" for m in modules):
                bindings = {a.asname or a.name.partition(".")[0] for a in stmt.names}
                if "*" not in bindings and not names.intersection(bindings):
                    imports.append(ast.dump(stmt))
                    names.update(bindings)
                    continue
            result.extend(sorted(imports))
            imports.clear()
            names.clear()
            result.append(ast.dump(stmt))
        return result + sorted(imports)
    return normalized(original) == normalized(updated)


if __name__ == "__main__":
    build()
