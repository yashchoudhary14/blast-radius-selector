"""Apply unified diffs at their stated line numbers, then link the untouched tree."""

import ast
import os
import re


class PatchError(ValueError):
    pass


HEADER = re.compile(rb"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*|)$")
NO_EOL = b"\\ No newline at end of file"


def _path(raw):
    raw = raw.split(b"\t", 1)[0]
    if raw == b"/dev/null":
        return None
    if raw.startswith(b'"'):
        raw = ast.literal_eval("b" + raw.decode("ascii"))
    if raw.startswith((b"a/", b"b/")):
        raw = raw[2:]
    if any(part in (b"", b".", b"..") for part in raw.split(b"/")) or b"\0" in raw:
        raise PatchError(f"invalid patch path {raw!r}")
    return os.fsdecode(raw)


def _lines(raw):
    parts = raw.split(b"\n")
    return [line + b"\n" for line in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def _body(lines, i, count, new_count, name):
    rows = []
    while count or new_count:
        if i >= len(lines) or lines[i][:1] not in (b" ", b"-", b"+"):
            raise PatchError(f"{name}: incomplete hunk")
        kind, body = lines[i][:1], lines[i][1:]
        count -= kind != b"+"
        new_count -= kind != b"-"
        if count < 0 or new_count < 0:
            raise PatchError(f"{name}: hunk counts disagree")
        i += 1
        has_eol = i == len(lines) or lines[i] != NO_EOL
        rows.append((kind, body, has_eol))
        if not has_eol:
            i += 1
    return rows, i


def _apply(lines, i, source, name):
    old = _lines(source)
    out = []
    cursor = 0
    crlf = source.count(b"\r\n") > source.count(b"\n") / 2
    while i < len(lines) and lines[i].startswith(b"@@"):
        m = HEADER.fullmatch(lines[i])
        if not m:
            raise PatchError(f"{name}: bad hunk header {lines[i]!r}")
        start, count, new_start, new_count = (int(x or 1) for x in m.groups())
        pos = start - bool(count)
        if not cursor <= pos <= len(old):
            raise PatchError(f"{name}: hunk outside source or out of order")
        out.extend(old[cursor:pos])
        if len(out) != new_start - bool(new_count):
            raise PatchError(f"{name}: incorrect target coordinates")
        cursor = pos
        rows, i = _body(lines, i + 1, count, new_count, name)
        # A git diff can carry literal CRs on hunk lines. An LF-normalized
        # diff against a CRLF checkout instead inherits the source convention.
        explicit_cr = any(body.endswith(b"\r") for _, body, _ in rows)
        for kind, body, terminated in rows:
            if kind != b"+":
                wanted = body + (b"\n" if terminated else b"")
                actual = old[cursor] if cursor < len(old) else None
                if actual and not explicit_cr and actual.endswith(b"\r\n"):
                    actual = actual[:-2] + b"\n"
                if actual != wanted:
                    raise PatchError(f"{name}: context mismatch at source line {cursor + 1}")
                if kind == b" ":
                    out.append(old[cursor])
                cursor += 1
            elif terminated:
                out.append(body + (b"\r\n" if crlf and not explicit_cr else b"\n"))
            else:
                out.append(body)
    out.extend(old[cursor:])
    return b"".join(out), i


def changes(base, patch):
    raw = patch.encode("utf-8", "surrogateescape")
    if raw.partition(b"\n")[0].endswith(b"\r"):
        raw = raw.replace(b"\r\n", b"\n")
    lines = raw.split(b"\n")
    edits = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if not line or line.startswith((b"index ", b"new file mode ", b"deleted file mode ")):
            continue
        if line.startswith(b"diff --git "):
            # Git omits ---/+++ for creation/deletion of an empty file.
            if i < len(lines) and lines[i].startswith((b"new file mode ", b"deleted file mode ")):
                j = i + 1
                if j < len(lines) and lines[j].startswith(b"index "):
                    j += 1
                if j == len(lines) or not lines[j] or lines[j].startswith(b"diff --git "):
                    left, sep, right = line[11:].rpartition(b" b/")
                    if not sep or _path(left) != _path(b"b/" + right):
                        raise PatchError("unsupported empty file header")
                    path = _path(left)
                    creating = lines[i].startswith(b"new")
                    if creating == (base / path).exists() or (
                        not creating and (base / path).stat().st_size
                    ):
                        raise PatchError(f"{path}: invalid empty file change")
                    edits[path] = b"" if creating else None
            continue
        if not line.startswith(b"--- ") or i >= len(lines) or not lines[i].startswith(b"+++ "):
            raise PatchError(f"unsupported patch header {line[:120]!r}")
        old, new = _path(line[4:]), _path(lines[i][4:])
        i += 1
        if old is None and new is None:
            raise PatchError("missing file name")
        try:
            if old is not None and not (base / old).resolve().is_relative_to(base):
                raise PatchError(f"{old}: source outside base")
            if old != new and new and (base / new).exists():
                raise PatchError(f"{new}: target already exists")
            if old in edits or new in edits:
                raise PatchError("repeated file section")
            source = (base / old).read_bytes() if old else b""
            if i >= len(lines) or not lines[i].startswith(b"@@"):
                raise PatchError(f"{new or old}: missing hunk")
            result, i = _apply(lines, i, source, new or old)
            if new is None and result:
                raise PatchError(f"{old}: incomplete deletion")
            if old and old != new:
                edits[old] = None
            if new:
                edits[new] = result
        except OSError as exc:
            raise PatchError(str(exc)) from exc
    return edits


def populate(base, root, edits):
    # This pinned test resolves its own __file__ to locate sqlglotc/setup.py.
    # Give it an overlay inode if that script or its package changes.
    anchor = "tests/test_sqlglotc.py"
    if (any(p.startswith("sqlglotc/") for p in edits) and anchor not in edits
            and (base / anchor).is_file()):
        edits = {**edits, anchor: (base / anchor).read_bytes()}
    tree = {}
    for name, data in edits.items():
        directory = tree
        *parents, leaf = name.split("/")
        for part in parents:
            directory = directory.setdefault(part, {})
        directory[leaf] = data

    def visit(src, dst, entries):
        dst.mkdir()
        # A checked-out source may have the same timestamp and size as its
        # pristine version. Never expose its old pyc, even an unchecked-hash pyc.
        changed_py = {name[:-3] for name in entries if name.endswith(".py")}
        cache = src / "__pycache__"
        if changed_py and cache.is_dir():
            omitted = entries.setdefault("__pycache__", {})
            for child in cache.iterdir():
                if child.name.split(".", 1)[0] in changed_py:
                    omitted.setdefault(child.name, None)
        if src.is_dir():
            for child in os.listdir(src):
                if child not in entries:
                    os.symlink(src / child, dst / child)
        for child, data in entries.items():
            path = dst / child
            if isinstance(data, dict):
                visit(src / child, path, data)
            elif data is not None:
                path.write_bytes(data)

    visit(base, root, tree)
