"""Trace reads of the generator's deferred modifier callbacks."""

import ast
import re
from functools import lru_cache

from .source_equivalence import _same_directives_text_only


RENDER_MODIFIERS = "AFTER_HAVING_MODIFIER_TRANSFORMS"
TABLE_SOURCE = re.compile(rb"(?ms)^ *AFTER_HAVING_MODIFIER_TRANSFORMS:[^\n]*\n.*?^ *\}")


def _ordered_segments(mapping):
    segments, entries = [], {}
    for key, value in zip(mapping.keys, mapping.values):
        if key is None:
            segments.append((entries, ast.dump(value)))
            entries = {}
        else:
            entries[ast.dump(key)] = ast.dump(value)
    return segments, entries


def render_modifier_edit(before, after, path, *, order_only=False):
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
        if order_only and _ordered_segments(a.value) != _ordered_segments(b.value):
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


@lru_cache(maxsize=64)
def render_modifier_read_lines(analyzer, path, *, order_only=False):
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
        fixed_value = (order_only and isinstance(parent, ast.Subscript) and parent.value is node
                       and isinstance(parent.ctx, ast.Load) and isinstance(parent.slice, ast.Constant)
                       and type(parent.slice.value) is str)
        (passive if copies or fixed_value else active).update(range(node.lineno, node.end_lineno + 1))
    return set(refs) - (passive - active)


@lru_cache(maxsize=2)
def modifier_order_readers(analyzer):
    readers = analyzer.location_observers(all_dynamic=True) | analyzer.state_refs.mask("*module")
    for path in analyzer.state_refs.refs.get(RENDER_MODIFIERS, ()):
        readers |= analyzer._lines_and_indirect_mask(path, render_modifier_read_lines(analyzer, path, order_only=True))
    return readers
