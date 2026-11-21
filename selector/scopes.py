"""Structural signatures for module/class statements and entire function bodies."""

import ast
from collections import Counter
from hashlib import blake2b

FUNCTION = (ast.FunctionDef, ast.AsyncFunctionDef)


def digest(data):
    return blake2b(data, digest_size=16).digest()


def structural(node):
    return digest(ast.dump(node, include_attributes=False).encode("utf-8", "surrogatepass"))


def snapshot(tree, source, cached=None):
    """Structurally hash function headers and bodies using AST comparison."""
    functions, scopes = {}, {}

    def visit_nested_funcs(func_node, prefix):
        counts = Counter()
        saved = []
        for child in ast.walk(func_node):
            if child is not func_node and isinstance(child, FUNCTION):
                name = prefix + child.name
                counts[name] += 1
                if counts[name] > 1:
                    name += "#" + str(counts[name])
                start = min([child.lineno, *(d.lineno for d in child.decorator_list)])
                body = structural(child)
                saved_body = child.body
                child.body = []
                header = structural(child)
                child.body = [ast.Pass()]
                saved.append((child, saved_body))
                functions[name] = body, header, body, start, child.end_lineno
        return saved

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
            visit_nested_funcs(stmt, name + ".")
            start = min([stmt.lineno, *(d.lineno for d in stmt.decorator_list)])
            body = structural(stmt)
            stmt.body = []
            header = structural(stmt)
            functions[name] = body, header, body, start, stmt.end_lineno
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
