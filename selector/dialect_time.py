"""Attribute concrete time-format dictionaries to their pinned consumers."""

import ast
from functools import lru_cache

from .dialect_tables import concrete, named, same_directive_lines, target, transmitted


TIME = {
    "TIME_MAPPING", "TIME_TRIE", "INVERSE_TIME_MAPPING", "INVERSE_TIME_TRIE", "FORMAT_TRIE",
    "MYSQL_INVERSE_TIME_MAPPING", "MYSQL_INVERSE_TIME_TRIE",
}


def time_changes(path, before, after):
    if not concrete(path, "dialects") or b"TIME_MAPPING" not in before:
        return None
    if not same_directive_lines(before, after):
        return None
    old, new = ast.parse(before), ast.parse(after)

    def table(tree):
        declarations = [
            node for cls in tree.body if isinstance(cls, ast.ClassDef) and not cls.decorator_list and not cls.keywords
            for node in cls.body if isinstance(target(node), ast.Name) and target(node).id == "TIME_MAPPING"
        ]
        if len(declarations) != 1 or not isinstance(declarations[0].value, ast.Dict):
            return None
        return declarations[0]

    left, right = table(old), table(new)
    if left is None or right is None or ast.dump(left.value) == ast.dump(right.value):
        return None
    for stmt in (left, right):
        mapping = stmt.value
        if not mapping.keys or not all(isinstance(item, ast.Constant) and type(item.value) is str and item.value
                                       for item in (*mapping.keys, *mapping.values)):
            return None
    a, b = ast.literal_eval(left.value), ast.literal_eval(right.value)
    right.value = left.value
    if ast.dump(old) != ast.dump(new):
        return None
    return a.get("MM") != b.get("MM")


def passthrough(file, node, parents, field, strict_changed):
    if transmitted(node, parents, field):
        return True
    parent = parents[node]
    if named(target(parent)) in TIME and parent.value is node:
        return True
    if (file == "sqlglot/generators/hive.py" and not strict_changed and isinstance(parent, ast.Attribute)
            and parent.attr == "get" and ast.unparse(parents[parent]) == 'self.dialect.TIME_MAPPING.get(\'MM\')'):
        return True
    if file == "sqlglot/dialects/dialect.py":
        scope = node
        while scope in parents and not isinstance(scope, ast.FunctionDef):
            scope = parents[scope]
        if isinstance(scope, ast.FunctionDef) and scope.name == "__new__":
            # Only the metaclass builds new forward, inverse and format tries.
            return True
    if file == "sqlglot/parsers/tsql.py" and isinstance(parent, (ast.Dict, ast.IfExp)):
        # Optional expansion passed straight to format_time.
        parent = parents.get(parent) if isinstance(parent, ast.Dict) else parent
    while isinstance(parent, (ast.BoolOp, ast.IfExp, ast.keyword)):
        parent = parents.get(parent)
    return isinstance(parent, ast.Call) and named(parent.func) == "format_time"


@lru_cache(maxsize=4)
def time_readers(a, strict_changed):
    file = "sqlglot/time.py"
    tree, _ = a.old_tree_and_lines(file)
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "format_time")
    mask = a._lines_and_indirect_mask(file, range(fn.body[1].lineno, fn.end_lineno + 1))
    mask |= a.state_refs.mask("*") | a.doc_observers() | a.location_observers()
    for name in TIME:
        for path, lines in a.state_refs.refs.get(name, {}).items():
            if path.startswith("tests/"):
                mask |= a._lines_and_indirect_mask(path, lines)
                continue
            tree, _ = a.old_tree_and_lines(path)
            parents = {node: parent for parent in ast.walk(tree) for node in ast.iter_child_nodes(parent)}
            safe, unsafe = set(), set()
            for node in parents:
                if (path == "sqlglot/dialects/dialect.py" and isinstance(node, ast.Call)
                        and ast.unparse(node) == f"klass.__dict__.get({name!r})"
                        and passthrough(path, node, parents, name, strict_changed)):
                    safe.add(node.lineno)
                if not isinstance(node, (ast.Name, ast.Attribute)) or named(node) != name:
                    continue
                if isinstance(node.ctx, ast.Load):
                    hits = safe if passthrough(path, node, parents, name, strict_changed) else unsafe
                    hits.update((node.lineno, node.end_lineno))
            mask |= a._lines_and_indirect_mask(path, set(lines) - (safe - unsafe))
    return mask
