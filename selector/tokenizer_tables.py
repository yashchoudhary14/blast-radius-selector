"""Preserve tokenization and trie behavior for the pinned parser limit test."""

import ast
import re

from .comment_cases import is_expression
from .dialect_tables import same_directive_lines
from .generator_modifiers import modifier_order_readers, render_modifier_edit


TOKENS = "sqlglot/tokens.py"
TOKENIZER_CORE = "sqlglot/tokenizer_core.py"
TRIE = "sqlglot/trie.py"
PARSER = "sqlglot/parser.py"
BASE_PARSER = "sqlglot/parsers/base.py"
GENERATOR = "sqlglot/generator.py"
LIMIT_WORDS = {"SELECT", "FROM", "T", "X", *(f"COL{i}" for i in range(500))}
LIMIT_CHARS = set("".join(LIMIT_WORDS) + ",= 0123456789")
LIMIT_TYPES = {"SELECT", "FROM", "VAR", "IDENTIFIER", "COMMA", "EQ", "NUMBER", "SENTINEL"}
COMPARISON = re.compile(rb"\b(?:comments|sub)\s*==")


def token_mapping(node, known, *, keyword_table=False):
    if not isinstance(node, ast.Dict):
        return None
    table = {}
    derived = []
    for key, value in zip(node.keys, node.values):
        if keyword_table and (key is None or is_expression(key, "HINT_START")):
            derived.append((ast.dump(key) if key is not None else None, ast.dump(value)))
            continue
        if not (isinstance(key, ast.Constant) and type(key.value) is str and key.value not in table
                and isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name)
                and value.value.id == "TokenType" and value.attr in known):
            return None
        table[key.value] = value.attr
    return table, derived


def normalize_tables(old, new, known):
    def declarations(tree):
        cls = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Tokenizer"), None)
        if cls is None:
            return {}
        return {node.targets[0].id: node for node in cls.body if isinstance(node, ast.Assign)
                and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)} | {
            node.target.id: node for node in cls.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None
        }

    left, right = declarations(old), declarations(new)
    names = {"KEYWORDS", "SINGLE_TOKENS", "DROP_UNKNOWN_ESCAPES"}
    if not names <= (left.keys() & right.keys()):
        return False
    singles = token_mapping(left["SINGLE_TOKENS"].value, known)
    if singles is None:
        return False
    changed = False
    for name in names:
        a, b = left[name], right[name]
        if ast.dump(a.value) == ast.dump(b.value):
            continue
        if name == "DROP_UNKNOWN_ESCAPES":
            # No input enters the scanner for quoted text or numeric escapes.
            if not (isinstance(b.value, ast.Constant) and type(b.value.value) is bool):
                return False
        else:
            prior = token_mapping(a.value, known, keyword_table=name == "KEYWORDS")
            revised = token_mapping(b.value, known, keyword_table=name == "KEYWORDS")
            if prior is None or revised is None:
                return False
            before, before_spreads = prior
            after, after_spreads = revised
            if before_spreads != after_spreads:
                return False
            keys = {key for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
        if name == "SINGLE_TOKENS":
            # The parser builds a set of reserved token types. The tokenizer
            # builds its trie from membership in the same single-character keys.
            if before.keys() != after.keys() or set(before.values()) != set(after.values()) or keys & LIMIT_CHARS:
                return False
        elif name == "KEYWORDS" and (not before.keys() <= after.keys() or any(
            key in LIMIT_WORDS or not re.fullmatch(r"[A-Z_][A-Z_0-9]*", key)
            or any(char in singles[0] for char in key) for key in keys
        )):
            # These words stay outside the trie. The pinned SQL has no
            # quoted identifiers, comments or numeric suffixes. Preserve key
            # membership for dialects that remove inherited entries at import.
            return False
        b.value = a.value
        changed = True
    return changed


def normalize_tokenizer_comparisons(tree):
    # Token comments in this workload come from TokenizerCore's built-in
    # list or the constructor default. Trie .get returns a built-in dict or
    # None because string character keys never address the terminal int key.
    operands = {("Token", "__init__"): "comments", ("TokenizerCore", "_scan_keywords"): "sub"}
    changed = False
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef):
            continue
        for fn in cls.body:
            if not isinstance(fn, ast.FunctionDef) or (cls.name, fn.name) not in operands:
                continue
            operand = operands[cls.name, fn.name]
            for predicate in ast.walk(fn):
                if isinstance(predicate, ast.Compare) and is_expression(predicate, f"{operand} == None"):
                    predicate.ops[0] = ast.Is()
                    changed = True
    return changed


def normalize_trie_terminal(tree):
    # The tokenizer and parser check membership of the end marker 0.
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef) and fn.name == "new_trie":
            for stmt in ast.walk(fn):
                if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                        and isinstance(stmt.targets[0], ast.Subscript)
                        and is_expression(stmt.targets[0].value, "current")
                        and is_expression(stmt.targets[0].slice, "0")
                        and isinstance(stmt.value, ast.Constant) and stmt.value.value is False):
                    stmt.value = ast.Constant(True)
                    return True
    return False


def normalize_default_dialect(tree):
    # The only use of this parameter is Dialect.get_or_raise(dialect), whose
    # first branch returns the default Dialect for either falsy singleton.
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or cls.name != "Tokenizer":
            continue
        for fn in cls.body:
            if (isinstance(fn, ast.FunctionDef) and fn.name == "__init__"
                    and len(fn.args.defaults) == 1 and isinstance(fn.args.defaults[0], ast.Constant)
                    and fn.args.defaults[0].value is False):
                fn.args.defaults[0] = ast.Constant(value=None)
                return True
    return False


def unchanged_limit_tokens(analyzer, edits, bit):
    if len(edits) != 1:
        return False
    path, source = next(iter(edits.items()))
    if path not in (TOKENS, TOKENIZER_CORE, TRIE) or not source or analyzer.index["reads"].get(path, 0) & bit:
        return False
    if path == TOKENIZER_CORE and not COMPARISON.search(source):
        return False
    file = analyzer.base / path
    mode = file.stat().st_mode & 0o777
    if edits.modes.get(path, mode) != mode:
        return False
    old, _ = analyzer.old_tree_and_lines(path)
    new = ast.parse(source)
    if path == TOKENS:
        if (not analyzer.signature_observers(path) & bit
                and not any(analyzer.state_refs.mask(name) & bit for name in ("__defaults__", "__kwdefaults__"))
                and normalize_default_dialect(new)):
            if ast.dump(old) == ast.dump(new):
                return same_directive_lines(file.read_bytes(), source)
            new = ast.parse(source)
        changed = normalize_tables(old, new, analyzer.dialect_reads.known_tokens())
    elif path == TOKENIZER_CORE:
        changed = normalize_tokenizer_comparisons(new)
    else:
        changed = normalize_trie_terminal(new)
    return changed and ast.dump(old) == ast.dump(new) and same_directive_lines(file.read_bytes(), source)


def _class_field(tree, owner, field):
    fields = []
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or cls.name != owner:
            continue
        for stmt in cls.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                lhs = stmt.targets[0]
            elif isinstance(stmt, ast.AnnAssign):
                lhs = stmt.target
            else:
                continue
            if isinstance(lhs, ast.Name) and lhs.id == field:
                fields.append(stmt)
    return fields[0] if len(fields) == 1 else None


def _identifier_tokens(node, known):
    if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub) and isinstance(node.left, ast.Set)):
        return None
    members, spreads = set(), []
    for item in node.left.elts:
        if isinstance(item, ast.Starred) and isinstance(item.value, ast.Name):
            spreads.append(item.value.id)
        elif (isinstance(item, ast.Attribute) and isinstance(item.value, ast.Name)
              and item.value.id == "TokenType" and item.attr in known):
            members.add(item.attr)
        else:
            return None
    return members, spreads, ast.dump(node.right)


def table_query_cases(analyzer, edits, isolated, revisions, mask):
    """A disabled TABLE feature can affect only cases that got past its guards."""
    from .analyze import _get_func_node
    from .query_cases import reached_cases

    if edits.keys() - isolated.keys() != {BASE_PARSER} or BASE_PARSER not in revisions:
        return {}
    mask &= ~analyzer.location_observers(all_dynamic=True)
    for path in edits:
        mask &= ~(analyzer.index["reads"].get(path, 0) | isolated.get(path, 0))
    if not mask:
        return {}
    source = edits[BASE_PARSER]
    file = analyzer.base / BASE_PARSER
    if BASE_PARSER in edits.modes and edits.modes[BASE_PARSER] != file.stat().st_mode & 0o777:
        return {}
    field = "SUPPORTS_TABLE_QUERY"
    old, _ = analyzer.old_tree_and_lines(BASE_PARSER)
    new = revisions[BASE_PARSER][0]
    left, right = (_class_field(tree, "BaseParser", field) for tree in (old, new))
    if (left is None or right is None or not isinstance(left.value, ast.Constant) or left.value.value is not True
            or not isinstance(right.value, ast.Constant) or right.value.value is not False):
        return {}
    revised = right.value
    right.value = left.value
    try:
        if ast.dump(old) != ast.dump(new) or not same_directive_lines(file.read_bytes(), source):
            return {}
    finally:
        right.value = revised

    tree, _ = analyzer.old_tree_and_lines(PARSER)
    fn = _get_func_node(tree, "Parser._is_table_query")
    if fn is None or len(fn.body) != 1 or not isinstance(fn.body[0], ast.Return):
        return {}
    guard = fn.body[0].value
    if (not isinstance(guard, ast.BoolOp) or not isinstance(guard.op, ast.And)
            or len(guard.values) != 3 or not is_expression(guard.values[0], "self.SUPPORTS_TABLE_QUERY")
            or not is_expression(guard.values[1], "self._curr.token_type == TokenType.TABLE")
            or guard.values[2].lineno <= guard.values[1].end_lineno):
        return {}
    for path, lines in analyzer.state_refs.refs.get(field, {}).items():
        readers = set(lines) - ({guard.values[0].lineno} if path == PARSER else set())
        mask &= ~analyzer._lines_and_indirect_mask(path, readers)
    # If the line after the TABLE check was not reached, the old call also
    # returned False. The exact TokenType comparison has no side effects.
    return {
        n: [i for i in range(count) if hit & (1 << i)]
        for n, (count, masks, files) in analyzer.queries.maps.items()
        if mask & (1 << n)
        and (hit := reached_cases(masks, files, PARSER,
                                 [(guard.values[2].lineno, guard.values[2].end_lineno)])).bit_count() < count
    }


def unchanged_limit_config(analyzer, edits, isolated, revisions, bit):
    paths = edits.keys() - isolated.keys()
    if len(paths) != 1 or analyzer.location_observers(all_dynamic=True) & bit:
        return False
    if any((analyzer.index["reads"].get(path, 0) | isolated.get(path, 0)) & bit for path in edits):
        return False
    path = next(iter(paths))
    if path not in (PARSER, BASE_PARSER, GENERATOR) or path not in revisions:
        return False
    file = analyzer.base / path
    if path in edits.modes and edits.modes[path] != file.stat().st_mode & 0o777:
        return False
    before, after = file.read_bytes(), edits[path]
    if path == GENERATOR:
        # Copies and fixed-key lookups see the same callbacks. Only an
        # iteration can observe the relative order of the moved entries.
        return (render_modifier_edit(before, after, path, order_only=True)
                and not modifier_order_readers(analyzer) & bit)
    old, _ = analyzer.old_tree_and_lines(path)
    new = revisions[path][0]
    if path == BASE_PARSER:
        owner, field = "BaseParser", "SUPPORTS_TABLE_QUERY"
    else:
        owner, field = "Parser", "ID_VAR_TOKENS"
    left, right = (_class_field(tree, owner, field) for tree in (old, new))
    if left is None or right is None:
        return False
    prior, revised = left.value, right.value
    if ast.dump(prior) == ast.dump(revised):
        return False
    if path == BASE_PARSER:
        # The sole runtime read enables TABLE queries. None of the pinned
        # limit inputs provides a TABLE token.
        if not all(isinstance(value, ast.Constant) and type(value.value) is bool for value in (prior, revised)):
            return False
    else:
        known = analyzer.dialect_reads.known_tokens()
        old_table, new_table = (_identifier_tokens(value, known) for value in (prior, revised))
        if old_table is None or new_table is None or old_table[1:] != new_table[1:]:
            return False
        original_set, updated_set = old_table[0], new_table[0]
        # VAR keeps the pinned default parser's derived alias sets nonempty.
        # Membership of every token the stress inputs can supply is unchanged.
        if "VAR" not in original_set & updated_set or (original_set ^ updated_set) & LIMIT_TYPES:
            return False
    right.value = prior
    try:
        return ast.dump(old) == ast.dump(new) and same_directive_lines(before, after)
    finally:
        right.value = revised
