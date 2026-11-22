"""Preserve tokenization and trie behavior for the pinned parser limit test."""

import ast
import re

from .comment_cases import is_expression
from .dialect_tables import same_directive_lines


TOKENS = "sqlglot/tokens.py"
TOKENIZER_CORE = "sqlglot/tokenizer_core.py"
TRIE = "sqlglot/trie.py"
LIMIT_WORDS = {"SELECT", "FROM", "T", "X", *(f"COL{i}" for i in range(500))}
LIMIT_CHARS = set("".join(LIMIT_WORDS) + ",= 0123456789")
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
