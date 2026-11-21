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


def signature(patch):
    return tuple(line for line in patch.splitlines() if line.startswith((b"diff --git ", b"+", b"-")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tree", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--exclude", nargs="+", type=Path, default=[], help="existing patch directories")
    args = parser.parse_args()
    rng = random.Random(855731)
    pool = list(candidates(args.tree))
    root = args.output
    (root / "commits").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.manifest, root / "tests.json")
    rng.shuffle(pool)
    excluded = {signature(patch.read_bytes()) for directory in args.exclude for patch in directory.glob("*.patch")}
    seen, written = set(), 0
    for file, start, token in pool:
        if (file, start) in seen:
            continue
        seen.add((file, start))
        path = file.relative_to(args.tree).as_posix()
        raw = file.read_bytes().replace(b"\r\n", b"\n")
        after = rng.choice(FLIPS[token.decode()]).encode()
        revised = raw[:start] + after + raw[start + len(token):]
        compile(revised, path, "exec")
        diff = difflib.diff_bytes(difflib.unified_diff, raw.splitlines(True), revised.splitlines(True),
                                  ("a/" + path).encode(), ("b/" + path).encode())
        patch = f"diff --git a/{path} b/{path}\n".encode() + b"".join(diff)
        fingerprint = signature(patch)
        if fingerprint in excluded:
            continue
        excluded.add(fingerprint)
        (root / "commits" / f"r{written:03d}.patch").write_bytes(patch)
        written += 1
        if written == 300:
            break
    print(f"{written} distinct mutants from {len(pool)} eligible token locations")
    if written != 300:
        raise SystemExit("insufficient distinct mutants")


if __name__ == "__main__":
    main()
