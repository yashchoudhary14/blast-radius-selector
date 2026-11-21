"""Sample distinct single-operator substitutions uniformly by site for the offline gate."""
import argparse
import ast
import difflib
import json
import random
import re
import shutil
from pathlib import Path

FLIPS = {"==": ("!=", "is"), "!=": ("==", "is not"), "is": ("==", "is not"),
         "is not": ("!=", "is"), "in": ("not in",), "not in": ("in",),
         "<": ("<=", ">"), "<=": ("<", ">="), ">": (">=", "<"), ">=": (">", "<="),
         "and": ("or",), "or": ("and",)}


def candidates(tree):
    for file in sorted((tree / "sqlglot").rglob("*.py")):
        raw = file.read_bytes().replace(b"\r\n", b"\n")
        starts = [0, *(m.end() for m in re.finditer(b"\n", raw))]
        for node in ast.walk(ast.parse(raw)):
            if isinstance(node, ast.Compare):
                values = [node.left, *node.comparators]
            elif isinstance(node, ast.BoolOp):
                values = node.values
            else:
                continue
            for a, b in zip(values, values[1:]):
                start = starts[a.end_lineno - 1] + a.end_col_offset
                end = starts[b.lineno - 1] + b.col_offset
                part = raw[start:end]
                token = part.strip()
                if token.decode() in FLIPS:
                    yield file, start + part.index(token), token


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tree", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rng = random.Random(855731)
    pool = list(candidates(args.tree))
    root = args.output
    (root / "commits").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.manifest, root / "tests.json")
    rng.shuffle(pool)
    seen = set()
    for file, start, token in pool:
        if (file, start) in seen:
            continue
        path = file.relative_to(args.tree).as_posix()
        raw = file.read_bytes().replace(b"\r\n", b"\n")
        after = rng.choice(FLIPS[token.decode()]).encode()
        revised = raw[:start] + after + raw[start + len(token):]
        compile(revised, path, "exec")
        diff = difflib.diff_bytes(difflib.unified_diff, raw.splitlines(True), revised.splitlines(True),
                                  ("a/" + path).encode(), ("b/" + path).encode())
        key = f"r{len(seen):03d}"
        (root / "commits" / (key + ".patch")).write_bytes(
            f"diff --git a/{path} b/{path}\n".encode() + b"".join(diff))
        seen.add((file, start))
        if len(seen) == 300:
            break
    print(f"{len(seen)} distinct mutants from {len(pool)} eligible token locations")


if __name__ == "__main__":
    main()
