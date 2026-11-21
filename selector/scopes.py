"""Structural signatures for module/class statements and entire function bodies."""

import ast
from collections import Counter
from hashlib import blake2b
from io import BytesIO
from tokenize import detect_encoding

FUNCTION = (ast.FunctionDef, ast.AsyncFunctionDef)


def digest(data):
    return blake2b(data, digest_size=16).digest()


def structural(node):
    return digest(ast.dump(node, include_attributes=False).encode("utf-8", "surrogatepass"))


def snapshot(tree, source, cached=None):
    """Only structurally hash function bodies whose exact source changed."""
    cached = cached or {}
    # An encoding declaration can change string literals anywhere in the
    # module while leaving the function's own bytes exactly the same.
    normalized = source.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    encoding, _ = detect_encoding(BytesIO(normalized).readline)
    lines = normalized.split(b"\n")
    functions, scopes = {}, {}

    def visit(node, prefix):
        counts = Counter()
        data, order = [], []
        for stmt in node.body:
            if not isinstance(stmt, (*FUNCTION, ast.ClassDef)):
                data.append(structural(stmt))
                order.append(None)
                continue
            name = prefix + stmt.name
            counts[name] += 1
            if counts[name] > 1:
                name += "#" + str(counts[name])
            order.append(name)
            if isinstance(stmt, ast.ClassDef):
                visit(stmt, name + ".")
                continue
            start = min([stmt.lineno, *(d.lineno for d in stmt.decorator_list)])
            raw = digest(encoding.encode() + b"\0" + b"\n".join(lines[start - 1:stmt.end_lineno]))
            old = cached.get(name)
            if old and raw == old[0]:
                header, body = old[1:3]
            else:
                body = structural(stmt)
                stmt.body = []
                header = structural(stmt)
            functions[name] = raw, header, body, start, stmt.end_lineno
        node.body = []
        scopes[prefix.rstrip(".")] = structural(node), tuple(data), tuple(order)

    visit(tree, "")
    return functions, scopes


def scope_changed(before, after):
    if before is None or after is None or before[:2] != after[:2]:
        return True
    if before[2] == after[2]:
        return False
    # Adding a method alone changes dispatch. Reordering existing definitions
    # relative to other definitions or class attributes can also change bindings.
    common = set(before[2]).intersection(after[2])
    return [k for k in before[2] if k in common] != [k for k in after[2] if k in common]
