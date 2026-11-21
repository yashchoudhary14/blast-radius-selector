"""Trace reads of the generator's deferred modifier callbacks."""

import ast
import re

from .source_equivalence import _same_directives_text_only


RENDER_MODIFIERS = "AFTER_HAVING_MODIFIER_TRANSFORMS"
TABLE_SOURCE = re.compile(rb"(?ms)^ *AFTER_HAVING_MODIFIER_TRANSFORMS:[^\n]*\n.*?^ *\}")


def render_modifier_edit(before, after, path):
    """Recognize changes confined to the generator's deferred callbacks."""
    if path != "sqlglot/generator.py" or TABLE_SOURCE.findall(before) == TABLE_SOURCE.findall(after):
        return False
    old, new = ast.parse(before), ast.parse(after)

    def tables(tree):
        cls = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Generator"), None)
        if cls is None:
            return []
        return [node for scope in (tree, cls) for node in scope.body
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                and node.target.id == RENDER_MODIFIERS and node.value is not None]

    left, right = tables(old), tables(new)
    if len(left) != 2 or len(right) != 2:
        return False
    changed = False
    for a, b in zip(left, right):
        if ast.dump(a.value) == ast.dump(b.value):
            continue
        if not isinstance(b.value, ast.Dict) or not isinstance(a.value, ast.Dict):
            return False
        if [ast.dump(v) for k, v in zip(a.value.keys, a.value.values) if k is None] != [
            ast.dump(v) for k, v in zip(b.value.keys, b.value.values) if k is None
        ]:
            return False
        keys = set()
        for key, value in zip(b.value.keys, b.value.values):
            if key is None:
                continue
            if not (isinstance(key, ast.Constant) and type(key.value) is str and key.value not in keys
                    and isinstance(value, ast.Lambda) and not value.args.defaults
                    and not any(value.args.kw_defaults)):
                return False
            keys.add(key.value)
        b.value = a.value
        changed = True
    return changed and ast.dump(old) == ast.dump(new) and _same_directives_text_only(before, after)


def render_modifier_read_lines(analyzer, path):
    refs = analyzer.state_refs.refs.get(RENDER_MODIFIERS, {}).get(path, ())
    tree, _ = analyzer.old_tree_and_lines(path)
    parents = {node: parent for parent in ast.walk(tree) for node in ast.iter_child_nodes(parent)}
    passive, active = set(), set()
    for node, parent in parents.items():
        if not isinstance(node, (ast.Name, ast.Attribute)) or not isinstance(node.ctx, ast.Load):
            continue
        name = node.attr if isinstance(node, ast.Attribute) else node.id
        if name != RENDER_MODIFIERS:
            continue
        if isinstance(parent, ast.Dict) and any(key is None and value is node
                                               for key, value in zip(parent.keys, parent.values)):
            node_value, assignment = parent, parents[parent]
        else:
            node_value, assignment = node, parent
        copies = (isinstance(assignment, ast.AnnAssign) and isinstance(assignment.target, ast.Name)
                  and assignment.target.id == RENDER_MODIFIERS and assignment.value is node_value)
        if isinstance(assignment, ast.Assign):
            copies = (len(assignment.targets) == 1 and isinstance(assignment.targets[0], ast.Name)
                      and assignment.targets[0].id == RENDER_MODIFIERS and assignment.value is node_value)
        # Only the pinned module and dialect generator declarations copy the
        # mapping, leaving lookups and enumeration to later render calls.
        copies &= (path == "sqlglot/generator.py" or path.startswith("sqlglot/generators/"))
        (passive if copies else active).update(range(node.lineno, node.end_lineno + 1))
    return set(refs) - (passive - active)
