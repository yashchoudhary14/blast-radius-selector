"""Project inert configuration assignments out of SQLGlot class bodies."""

import ast
from collections import Counter
from copy import copy

from .scopes import snapshot


def _value(node):
    if isinstance(node, ast.Constant):
        return type(node.value) in (bool, int, float, str, type(None))
    if isinstance(node, ast.Tuple):
        return all(_value(child) for child in node.elts)
    return False


def _target(node):
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        return node.targets[0]
    if isinstance(node, ast.AnnAssign):
        return node.target
    return None


def _declaration(node, postponed):
    name = _target(node)
    if not isinstance(name, ast.Name) or not name.id.isupper() or name.id.startswith("_"):
        return None
    if not node.value or not _value(node.value):
        return None
    if isinstance(node, ast.AnnAssign) and (not postponed or not node.simple):
        return None
    return name.id


def configuration_changes(path, old, new, old_functions, new_functions):
    """Report scalar class changes when other edits only modify function bodies."""
    if (old_functions.keys() != new_functions.keys()
            or any(row[1] != new_functions[name][1] for name, row in old_functions.items())):
        return set()

    def declarations(tree):
        postponed = any(isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__"
                        and any(name.name == "annotations" for name in stmt.names) for stmt in tree.body)
        values = {}
        for cls in tree.body:
            if not isinstance(cls, ast.ClassDef) or cls.decorator_list:
                continue
            if (cls.keywords and not (path == "sqlglot/dialects/dialect.py" and cls.name == "Dialect")):
                continue
            names = Counter(target.id for stmt in cls.body
                            if isinstance(target := _target(stmt), ast.Name))
            for i, node in enumerate(cls.body):
                if not (name := _declaration(node, postponed)) or names[name] != 1:
                    continue
                nodes = [node]
                following = cls.body[i + 1:i + 2]
                if (following and isinstance(following[0], ast.Expr)
                        and isinstance(following[0].value, ast.Constant)
                        and type(following[0].value.value) is str):
                    # An attribute docstring is a discarded constant statement.
                    nodes.append(following[0])
                values[cls.name, name] = tuple(nodes)
        return values

    def project(tree, assignments):
        remove = {node for key in changed for node in assignments.get(key, ())}
        projected = copy(tree)
        projected.body = []
        for cls in tree.body:
            if isinstance(cls, ast.ClassDef):
                doc = ast.get_docstring(cls, clean=False)
                cls = copy(cls)
                cls.body = [node for node in cls.body if node not in remove]
                if ast.get_docstring(cls, clean=False) != doc:
                    return None
            projected.body.append(cls)
        return snapshot(projected, b"", scopes_only=True)[1]

    left, right = declarations(old), declarations(new)
    changed = {key for key in left.keys() | right.keys()
               if tuple(map(ast.dump, left.get(key, ()))) != tuple(map(ast.dump, right.get(key, ())))}
    if not changed:
        return set()
    before = project(old, left)
    after = project(new, right)
    if before is None or before != after:
        return set()
    return {name for _, name in changed}


def added_methods(old, new, *, allow_body_edits=False):
    """Accept ordinary methods whose definitions execute no new user code."""
    postponed = any(isinstance(node, ast.ImportFrom) and node.module == "__future__"
                    and any(name.name == "annotations" for name in node.names) for node in old.body)
    if not postponed:
        return set()
    classes = {node.name: node for node in old.body if isinstance(node, ast.ClassDef)}
    methods = set()
    for cls in new.body:
        if not isinstance(cls, ast.ClassDef) or cls.name not in classes or cls.decorator_list or cls.keywords:
            continue
        if ast.get_docstring(cls, clean=False) != ast.get_docstring(classes[cls.name], clean=False):
            return set()
        # An existing binding may be a descriptor or an alias, not a def.
        if allow_body_edits:
            definition_time = []
            for stmt in classes[cls.name].body:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    definition_time.extend(stmt.decorator_list)
                    definition_time.extend(d for d in (*stmt.args.defaults, *stmt.args.kw_defaults) if d is not None)
                else:
                    definition_time.append(stmt)
            bound = {node.id for stmt in definition_time for node in ast.walk(stmt)
                     if isinstance(node, ast.Name)}
        else:
            bound = {node.id for node in ast.walk(classes[cls.name]) if isinstance(node, ast.Name)}
        bound.update(node.name for node in classes[cls.name].body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
        retained = []
        for node in cls.body:
            if (not isinstance(node, ast.FunctionDef) or node.name in bound or node.name.startswith("__")
                    or node.decorator_list or getattr(node, "type_params", ())
                    or not all(value is None or _value(value)
                               for value in (*node.args.defaults, *node.args.kw_defaults))):
                retained.append(node)
                continue
            bound.add(node.name)
            methods.add(node.name)
        cls.body = retained
    if not methods:
        return set()
    if allow_body_edits:
        before, old_scopes = snapshot(old, b"")
        after, new_scopes = snapshot(new, b"")
        old_headers = {name: row[1] for name, row in before.items()}
        new_headers = {name: row[1] for name, row in after.items()}
        return methods if old_scopes == new_scopes and old_headers == new_headers else set()
    return methods if ast.dump(old) == ast.dump(new) else set()
