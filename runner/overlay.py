"""Apply unified diffs at their stated line numbers, then link the untouched tree."""

import ast
import os
import re
from pathlib import PurePosixPath


class PatchError(ValueError):
    def __init__(self, message, reason="patch-error"):
        super().__init__(message)
        self.reason = reason


HEADER = re.compile(rb"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*|)$")
NO_EOL = b"\\ No newline at end of file"
MODE_HEADERS = (b"old mode", b"new mode", b"new file mode", b"deleted file mode")


def _path(raw, *, prefix=True):
    raw = raw.split(b"\t", 1)[0]
    if raw == b"/dev/null":
        return None
    if raw.startswith(b'"'):
        raw = ast.literal_eval("b" + raw.decode("ascii"))
    if prefix and raw.startswith((b"a/", b"b/")):
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


class Edits(dict):
    def __init__(self):
        super().__init__()
        self.modes = {}
        self.removed_dirs = set()

    def finish_deletions(self, base, removed):
        for path in removed:
            self.setdefault(path, None)
        # Git tracks files, so removing the last file also removes the directory.
        parents = {p for name in removed for p in PurePosixPath(name).parents if p.parts}
        retained = {p for name, data in self.items() if data is not None
                    for p in PurePosixPath(name).parents}
        for parent in sorted(parents - retained, key=lambda p: -len(p.parts)):
            if all(child.name == "__pycache__" or child.relative_to(base).as_posix() in removed
                   or child.relative_to(base).as_posix() in self.removed_dirs
                   for child in (base / parent).iterdir()):
                self.removed_dirs.add(parent.as_posix())


def _same_path(header):
    # Git does not quote ordinary spaces. In a metadata-only section both
    # names denote the same path, which disambiguates even an embedded " b/".
    match = re.fullmatch(rb'a/(.+) b/\1', header)
    if match:
        return _path(b"a/" + match[1])
    left, sep, right = header.partition(b'" "')
    if sep and _path(left + b'"') == _path(b'"' + right):
        return _path(left + b'"')
    raise PatchError(f"invalid file header {header!r}")


def changes(base, patch):
    raw = patch.encode("utf-8", "surrogateescape")
    if raw.partition(b"\n")[0].endswith(b"\r"):
        raw = raw.replace(b"\r\n", b"\n")
    lines = raw.split(b"\n")
    edits = Edits()
    removed = set()
    destinations = set()
    i = 0
    while i < len(lines):
        if not lines[i]:
            i += 1
            continue
        header = None
        meta = {}
        if lines[i].startswith(b"diff --git "):
            header = lines[i][11:]
            i += 1
            while i < len(lines) and lines[i] and not lines[i].startswith((b"diff --git ", b"--- ")):
                key, _, value = lines[i].rpartition(b" ")
                if value == b"120000" and (key in MODE_HEADERS or lines[i].startswith(b"index ")):
                    # Passing link target bytes to the regular-file overlay can
                    # make a probe pass even when the actual link breaks a test.
                    raise PatchError("symlink patches are unsupported", "symlink-patch")
                if lines[i].startswith((b"rename ", b"copy ")):
                    operation, key, value = lines[i].split(b" ", 2)
                    meta[b"operation"] = operation
                    meta[key] = _path(value, prefix=False)
                elif key in MODE_HEADERS:
                    meta[key] = int(value, 8)
                elif not lines[i].startswith((b"index ", b"similarity index ", b"dissimilarity index ")):
                    reason = "binary-patch" if lines[i].startswith((b"GIT binary patch", b"Binary files ")) else "patch-error"
                    raise PatchError(f"unsupported patch header {lines[i][:120]!r}", reason)
                i += 1

        hunks = i < len(lines) and lines[i].startswith(b"--- ")
        if hunks:
            if i + 1 >= len(lines) or not lines[i + 1].startswith(b"+++ "):
                raise PatchError("missing new file header")
            old, new = _path(lines[i][4:]), _path(lines[i + 1][4:])
            i += 2
        elif b"from" in meta and b"to" in meta:
            old, new = meta[b"from"], meta[b"to"]
        elif header is not None and meta:
            path = _same_path(header)
            old = None if b"new file mode" in meta else path
            new = None if b"deleted file mode" in meta else path
        else:
            raise PatchError(f"expected a file section near line {i + 1}")

        if not (old or new):
            raise PatchError("missing file name")
        try:
            if old and (base / old).is_symlink():
                raise PatchError("symlink patches are unsupported", "symlink-patch")
            if any(not (base / p).resolve().is_relative_to(base) for p in (old, new) if p):
                raise PatchError("path outside checkout")
            source = (base / old).read_bytes() if old else b""
            result = source
            if hunks:
                if i >= len(lines) or not lines[i].startswith(b"@@"):
                    raise PatchError(f"{new or old}: missing hunk")
                result, i = _apply(lines, i, source, new or old)
            if not new and result:
                raise PatchError(f"{old}: incomplete deletion")
            if old and old != new and meta.get(b"operation") != b"copy":
                removed.add(old)
            if new:
                if old != new:
                    destinations.add(new)
                if new in edits:
                    raise PatchError(f"{new}: repeated destination")
                edits[new] = result
                if b"new mode" in meta or b"new file mode" in meta:
                    edits.modes[new] = meta.get(b"new mode", meta.get(b"new file mode")) & 0o777
                elif old:
                    edits.modes[new] = (base / old).stat().st_mode & 0o777
        except OSError as exc:
            raise PatchError(str(exc)) from exc
    if removed:
        edits.finish_deletions(base, removed)
    for path in destinations - removed - edits.removed_dirs:
        if (base / path).exists():
            raise PatchError(f"{path}: target already exists")
    return edits


def populate(base, root, edits):
    # This pinned test resolves its own __file__ to locate sqlglotc/setup.py.
    # Give it an overlay inode if that script or its package changes.
    modes = getattr(edits, "modes", {})
    removed_dirs = getattr(edits, "removed_dirs", set())
    anchor = "tests/test_sqlglotc.py"
    if (any(p.startswith("sqlglotc/") for p in edits) and anchor not in edits
            and (base / anchor).is_file()):
        edits = {**edits, anchor: (base / anchor).read_bytes()}
    tree = {}
    # Place directory and file removals first so a new file may replace an
    # entire directory, or become a child of a path that used to be a file.
    deleted = removed_dirs | {p for p, data in edits.items() if data is None}
    staged = dict.fromkeys(sorted(deleted, key=lambda p: -p.count("/")))
    staged.update((p, data) for p, data in edits.items() if data is not None)
    for name, data in staged.items():
        directory = tree
        *parents, leaf = name.split("/")
        for part in parents:
            if directory.get(part) is None:
                directory[part] = {}
            directory = directory[part]
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
                mode = modes.get(path.relative_to(root).as_posix())
                if mode is not None:
                    path.chmod(mode)

    visit(base, root, tree)
