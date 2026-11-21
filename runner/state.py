"""Compare the import-time object graph in isolated, production-only interpreters."""
import enum
import functools
import hashlib
import importlib
import marshal
import re
import sys
import types
from pathlib import Path


BASELINE = Path("/app/maps/import-state.bin")
SCALAR = (type(None), bool, int, str, bytes, float, complex)
SKIP = {"__loader__", "__spec__", "__cached__", "__file__", "__path__", "__builtins__",
        "__dict__", "__weakref__", "__firstlineno__", "__warningregistry__"}


class ImportFailure(Exception):
    pass


def name(obj):
    return obj.__module__ + "." + obj.__qualname__


def code_parts(code, encode):
    return (code.co_code, tuple(encode(c) for c in code.co_consts), code.co_names,
            code.co_varnames, code.co_cellvars, code.co_freevars, code.co_exceptiontable,
            code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount,
            code.co_flags, code.co_name, code.co_qualname)


class Graph:
    """Intern structural nodes. Shared class objects are named edges to separate scopes."""

    def __init__(self, nodes):
        self.nodes = nodes
        self.ids = {node: i for i, node in enumerate(nodes)}
        self.memo = {}
        self.roots = {}
        self.classes = {}
        self.class_names = {}
        self.data = {}
        self.cycle = self.intern(("cycle",))

    def intern(self, node):
        i = self.ids.get(node)
        if i is None:
            i = self.ids[node] = len(self.nodes)
            self.nodes.append(node)
        return i

    def ref(self, obj):
        typ = type(obj)
        if typ in SCALAR:
            # Marshal retains signed zero, scalar type, and arbitrary integer widths.
            return self.intern(("atom", marshal.dumps(obj, 2)))
        cached = self.memo.get(id(obj))
        if cached is not None:
            return cached[1]
        self.memo[id(obj)] = (obj, self.cycle)
        i = self.intern(self.describe(obj))
        self.memo[id(obj)] = (obj, i)
        return i

    def describe(self, obj):
        typ = type(obj)
        ref = self.ref
        if isinstance(obj, type):
            ident = self.class_names.get(obj) or name(obj)
            if ident.startswith("sqlglot."):
                self.classes[ident] = obj
            return ("class", ident)
        if isinstance(obj, enum.Enum):
            return ("enum", name(typ), obj.name, ref(obj.value))
        if isinstance(obj, types.ModuleType):
            return ("module", obj.__name__)
        if isinstance(obj, dict):
            return ("dict", name(typ), tuple((ref(k), ref(v)) for k, v in obj.items()))
        if isinstance(obj, (tuple, list, set, frozenset)):
            items = tuple(map(ref, obj))
            if isinstance(obj, (set, frozenset)):
                items = tuple(sorted(items))
            return ("items", name(typ), items)
        if isinstance(obj, types.CodeType):
            return ("code", *code_parts(obj, ref))
        if isinstance(obj, types.FunctionType):
            if not (obj.__module__ or "").startswith("sqlglot"):
                return ("external-function", obj.__module__, obj.__qualname__)
            cells = []
            for cell in obj.__closure__ or ():
                try:
                    cells.append(ref(cell.cell_contents))
                except ValueError:
                    cells.append(-1)
            return ("function", obj.__module__, obj.__qualname__, ref(obj.__code__),
                    ref(obj.__defaults__), ref(obj.__kwdefaults__), tuple(cells),
                    ref(obj.__annotations__), ref(vars(obj)))
        if isinstance(obj, (staticmethod, classmethod, types.MethodType)):
            return ("descriptor", name(typ), ref(obj.__func__),
                    ref(obj.__self__) if isinstance(obj, types.MethodType) else -1)
        if isinstance(obj, property):
            return ("property", ref(obj.fget), ref(obj.fset), ref(obj.fdel))
        if isinstance(obj, re.Pattern):
            return ("regex", ref(obj.pattern), obj.flags)
        if isinstance(obj, functools.partial):
            return ("partial", ref(obj.func), ref(obj.args), ref(obj.keywords))
        if hasattr(obj, "__wrapped__") and callable(obj):
            return ("wrapped", name(typ), ref(obj.__wrapped__))
        if typ.__module__.startswith("sqlglot"):
            attrs = vars(obj) if hasattr(obj, "__dict__") else {}
            slots = []
            for cls in typ.__mro__:
                for key, member in vars(cls).items():
                    if isinstance(member, types.MemberDescriptorType) and hasattr(obj, key):
                        slots.append((key, ref(getattr(obj, key))))
            return ("instance", name(typ), ref(attrs), tuple(slots))
        # Foreign implementations may hide state. Changes to opaque bindings
        # cannot justify name-level selection (see the selector's eligibility gate).
        return ("foreign", name(typ), re.sub(r"0x[0-9a-fA-F]+", "0x?", repr(obj)))

    def scope(self, owner, attrs):
        for key, value in attrs.items():
            if key not in SKIP:
                self.roots[owner, key] = self.ref(value)
                if type(value) in (dict, tuple, list, set, frozenset) or isinstance(value, types.FunctionType):
                    self.data[owner, key] = value

    def aliases(self):
        """Identify shared mutable containers/functions independently of equality.

        Paths use structural key IDs from the baseline dictionary. Visit the
        first path only; the alias edge represents the entire shared subtree.
        """
        seen, aliases = {}, {}
        def visit(value, root, path, edges):
            typ = type(value)
            if typ not in (dict, list, tuple, set, frozenset, types.FunctionType):
                return
            previous = seen.get(id(value))
            if previous is not None:
                if typ not in (tuple, frozenset):
                    edges.append((path, previous))
                return
            # Tuple sharing itself is immaterial here, but its children can be mutable.
            if typ not in (tuple, frozenset):
                seen[id(value)] = root, path
            if typ is dict:
                children = [(self.ref(k), v) for k, v in value.items()
                            if type(v) in (dict, list, tuple, set, frozenset, types.FunctionType)]
                for key, child in sorted(children, key=lambda pair: pair[0]):
                    visit(child, root, path + (key,), edges)
            elif typ in (list, tuple):
                for i, child in enumerate(value):
                    visit(child, root, path + (i,), edges)
            elif typ in (set, frozenset):
                for child in sorted(value, key=self.ref):
                    visit(child, root, path + (self.ref(child),), edges)
        for root, value in sorted(self.data.items()):
            edges = []
            visit(value, root, (), edges)
            if edges:
                aliases[root] = hashlib.blake2b(marshal.dumps(edges, 2), digest_size=16).digest()
        return aliases

    def capture(self, root):
        base = Path("/sqlglot")
        for path in sorted((base / "sqlglot").rglob("*.py")):
            if path.name != "__main__.py":
                parts = path.relative_to(base).with_suffix("").parts
                try:
                    importlib.import_module(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
                except (Exception, SystemExit) as exc:
                    raise ImportFailure(str(path)) from exc
        # type("Tokenizer", ...) creates many distinct classes with the same
        # qualname. Their owning dialect provides a stable identity and exposes
        # derived attributes that do not occur as bindings on any module.
        from sqlglot import Dialect
        for dialect in Dialect.classes.values():
            for attr in ("tokenizer_class", "jsonpath_tokenizer_class", "parser_class", "generator_class"):
                cls = getattr(dialect, attr)
                if name(cls) in ("sqlglot.dialects.dialect.Tokenizer", "sqlglot.dialects.dialect.JSONPathTokenizer"):
                    self.class_names.setdefault(cls, name(dialect) + "." + attr)
        for mod, value in sorted(sys.modules.copy().items()):
            if mod == "sqlglot" or mod.startswith("sqlglot."):
                if not Path(value.__file__).is_relative_to(root):
                    raise ValueError(f"import outside overlay: {mod}")
                self.scope(mod, vars(value))
        seen = set()
        while pending := self.classes.keys() - seen:
            for ident in sorted(pending):
                cls = self.classes[ident]
                seen.add(ident)
                self.scope(ident, vars(cls))
                self.roots[ident, "__bases__"] = self.ref(cls.__bases__)
                self.roots[ident, "__mro__"] = self.ref(cls.__mro__)
        return self.roots


def diff(root, *, calibrate=False, details=False):
    with BASELINE.open("rb") as source:
        nodes, original, unordered, old_aliases = marshal.load(source)
    graph = Graph(nodes)
    updated = graph.capture(root)
    aliases = graph.aliases()
    result = {}
    for key in original.keys() | updated.keys():
        old, new = original.get(key), updated.get(key)
        if old != new:
            # Some derived tables iterate a set of classes, whose hashes depend
            # on their addresses. Calibrate just those roots in a second pristine
            # interpreter; their keys, values and multiplicities still count.
            if (calibrate or key in unordered) and old is not None and new is not None:
                a, b = nodes[old], nodes[new]
                if (a[0] in ("dict", "items") and a[:2] == b[:2]
                        and sorted(a[2]) == sorted(b[2])):
                    unordered.add(key)
                    continue
            paths = changed_paths(nodes, old, new)
            entries = None if () in paths else sorted({p[0] for p in paths})
            kind, keyed = "other", False
            if old is not None and new is not None:
                a, b = nodes[old], nodes[new]
                if a[0] in ("atom", "dict", "items", "regex") and b[0] in ("atom", "dict", "items", "regex"):
                    kind = "class" if key[0] in graph.classes else "module"
                if a[0] == b[0] == "dict" and entries:
                    keyed = all(nodes[k][0] == "atom" and type(marshal.loads(nodes[k][1])) is str
                                for pairs in (a[2], b[2]) for k, _ in pairs)
            result[key] = paths if details else (entries, kind, keyed)
    for key in aliases.keys() | old_aliases.keys():
        if aliases.get(key) != old_aliases.get(key):
            # A structurally identical shallow copy can be observed with 'is'
            # or through subsequent mutation, even in an otherwise empty diff.
            result[key] = [()] if details else (None, "other", False)
    if calibrate:
        if result:
            raise ValueError(f"unstable production snapshot: {list(result)[:8]}")
        with BASELINE.open("wb") as target:
            marshal.dump((nodes, original, unordered, old_aliases), target)
    return result


def changed_paths(nodes, old, new):
    """Relative paths to the first changed leaf, including container order."""
    if old is None or new is None:
        return [()]
    left, right = nodes[old], nodes[new]
    if left[:2] != right[:2]:
        return [()]
    def label(k):
        node = nodes[k]
        if node[0] == "atom":
            return str(marshal.loads(node[1]))
        if node[0] == "enum":
            return node[1] + "." + node[2]
        return str(node[1]) if node[0] == "class" else repr(node)
    if left[0] == "dict":
        a, b = dict(left[2]), dict(right[2])
        paths = [()] if [k for k in a if k in b] != [k for k in b if k in a] else []
        for k in a.keys() | b.keys():
            if a.get(k) != b.get(k):
                paths.extend((label(k), *tail) for tail in changed_paths(nodes, a.get(k), b.get(k)))
        return paths
    elif left[0] == "items" and left[1] in ("builtins.set", "builtins.frozenset"):
        return [(label(k),) for k in set(left[2]) ^ set(right[2])]
    elif left[0] == "items":
        from itertools import zip_longest
        return [(str(i), *tail) for i, (a, b) in enumerate(zip_longest(left[2], right[2]))
                if a != b for tail in changed_paths(nodes, a, b)]
    return [()]


if __name__ == "__main__":
    if "--calibrate" in sys.argv:
        diff(Path("/sqlglot"), calibrate=True)
    else:
        graph = Graph([])
        graph.capture(Path("/sqlglot"))
        with BASELINE.open("wb") as target:
            marshal.dump((graph.nodes, graph.roots, set(), graph.aliases()), target)
