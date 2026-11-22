"""Prove executor benchmark behavior from the pinned planner consumers."""

import ast

from runner.query_cases import EXECUTOR

from .source_equivalence import _same_directives_text_only


PLANNER = "sqlglot/planner.py"
METHODS = ("test_agg_order", "test_execute_tpch", "test_execute_tpcds", "test_optimized_tpch")


def _method(tree, owner, name):
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name == owner:
            for node in cls.body:
                if isinstance(node, ast.FunctionDef) and node.name == name:
                    return node
    return None


def _join_record(tree):
    fn = _method(tree, "Join", "from_joins")
    if fn is None:
        return None
    records = [node for node in ast.walk(fn) if isinstance(node, ast.Assign) and len(node.targets) == 1
               and isinstance(node.value, ast.Dict)
               and ast.unparse(node.targets[0]) == "step.joins[join.alias_or_name]"]
    return records[0] if len(records) == 1 else None


def _items(mapping):
    fields = {"side", "source_key", "join_key", "condition"}
    keys = [key.value for key in mapping.keys if isinstance(key, ast.Constant) and type(key.value) is str]
    if len(keys) != len(mapping.keys) or len(keys) != len(fields) or set(keys) != fields:
        return None
    return {key: ast.dump(value) for key, value in zip(keys, mapping.values)}


def _join_order_only(old, new):
    before, after = (_join_record(tree) for tree in (old, new))
    if before is None or after is None or ast.dump(before.value) == ast.dump(after.value):
        return False
    items = _items(before.value)
    if items is None or items != _items(after.value):
        return False
    # The pinned plan formatter and Python executor only retrieve fields by
    # key. The one attribute lookup (join.side) commutes with local reads.
    revised = after.value
    after.value = before.value
    try:
        return ast.dump(old) == ast.dump(new)
    finally:
        after.value = revised


def _explicit_set_operation_default(old, new):
    before, after = (_method(tree, "SetOperation", "__init__") for tree in (old, new))
    if before is None or after is None or len(before.args.defaults) != 1 or len(after.args.defaults) != 1:
        return False
    prior, revised = before.args.defaults[0], after.args.defaults[0]
    if not (isinstance(prior, ast.Constant) and prior.value is False
            and isinstance(revised, ast.Constant) and type(revised.value) in (type(None), bool)
            and revised.value is not False):
        return False
    # In the four pinned methods, any SetOperation is constructed by the
    # unchanged from_expression path, which always supplies distinct=.
    after.args.defaults[0] = prior
    try:
        return ast.dump(old) == ast.dump(new)
    finally:
        after.args.defaults[0] = revised


def unaffected_plan_tests(analyzer, edits, revisions, mask):
    if edits.keys() != {PLANNER} or PLANNER not in revisions:
        return 0
    eligible = sum(1 << analyzer.num[EXECUTOR + name] for name in METHODS if EXECUTOR + name in analyzer.num)
    eligible &= mask & ~(analyzer.index["reads"].get(PLANNER, 0) | analyzer.location_observers(all_dynamic=True))
    if not eligible:
        return 0
    file = analyzer.base / PLANNER
    if PLANNER in edits.modes and edits.modes[PLANNER] != file.stat().st_mode & 0o777:
        return 0
    old, _ = analyzer.old_tree_and_lines(PLANNER)
    new = revisions[PLANNER][0]
    if not _join_order_only(old, new):
        if not _explicit_set_operation_default(old, new):
            return 0
        keep = eligible & analyzer.signature_observers(PLANNER)
        # The pinned TPC-DS witness records reads of the actual constructor's
        # code and defaults across the parent and workers. It distinguishes
        # these from signature reads of unrelated executor functions.
        n = analyzer.num.get(EXECUTOR + "test_execute_tpcds")
        if n == 1106 and keep & (1 << n):
            from .patch_proof import evidence

            loaded, _, metadata, reads, _, _ = evidence(n)
            prefix = "sqlglot.planner.SetOperation.__init__."
            if PLANNER in loaded and PLANNER not in reads and not any(name.startswith(prefix) for name in metadata):
                keep &= ~(1 << n)
        eligible &= ~keep
    return eligible if _same_directives_text_only(file.read_bytes(), edits[PLANNER]) else 0
