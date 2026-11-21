"""Conservative phase-boundary proofs for the five pinned executor/optimizer tests."""
import bisect
import contextlib
import functools
import gzip
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

MAPS = Path(__file__).resolve().parents[1] / "maps/proxy"
TESTS = (1082, 1106, 1107, 1123, 1361)
# Child wall limits. A late, failed or interrupted proof retains the candidate.
LIMITS = {1082: 4, 1106: 12, 1107: 4, 1123: 5, 1361: 8}


@functools.lru_cache(maxsize=len(TESTS))
def protected(number):
    return json.loads(gzip.decompress((MAPS / f"{number}.guard.gz").read_bytes()))


def overlay_matches(root, paths):
    """Session overlays link every pristine entry and omit deleted entries."""
    base = Path("/sqlglot")
    if root == base:
        return not paths
    found = set()
    pending = [Path(".")]
    while pending:
        rel = pending.pop()
        original, staged = base / rel, root / rel
        if set(os.listdir(original)) - {"__pycache__"} != set(os.listdir(staged)) - {"__pycache__"}:
            return False
        for child in staged.iterdir():
            ref = original / child.name
            if child.name == "__pycache__":
                # The runner can expose an intentionally filtered directory of
                # symlinks to pristine bytecode. Reject any replacement bytecode.
                if child.is_symlink() and child.resolve() == ref.resolve():
                    continue
                if not child.is_dir() or any(not p.is_symlink() or p.resolve() != (ref / p.name).resolve()
                                             for p in child.iterdir()):
                    return False
            elif child.is_symlink() and child.resolve() == ref.resolve():
                continue
            elif not child.is_symlink() and child.is_dir() and ref.is_dir():
                pending.append(rel / child.name)
            elif not child.is_symlink() and child.is_file() and ref.is_file():
                found.add((rel / child.name).as_posix())
            else:
                return False
    return found == set(paths)


def can_prove_pass(test_number, changed_spans, data_changed_modules, overlay_root, *, deadline=None) -> bool | None:
    """Spans are inclusive old-line pairs keyed by every changed checkout-relative path."""
    if test_number not in TESTS or changed_spans is None or data_changed_modules is None:
        return None
    try:
        if data_changed_modules:
            return None
        coverage = protected(test_number)
        paths = []
        for path, spans in changed_spans.items():
            if not isinstance(path, str) or not path.startswith("sqlglot/") or not path.endswith(".py"):
                return None
            if ".." in Path(path).parts or not spans:
                return None
            lines = coverage.get(path, ())
            for start, end in spans:
                if not 0 < start <= end:
                    return None
                i = bisect.bisect_left(lines, start)
                if i < len(lines) and lines[i] <= end:
                    return None
            paths.append(path)
        root = Path(overlay_root).resolve(strict=True)
        if not overlay_matches(root, paths):
            return None
        env = dict(os.environ, PYTHONPATH=str(root), PYTHONHASHSEED="0",
                   PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", SKIP_INTEGRATION="0")
        timeout = LIMITS[test_number]
        if deadline is not None:
            timeout = min(timeout, deadline - time.monotonic())
            if timeout <= 0:
                return None
        with subprocess.Popen([sys.executable, "-B", __file__, str(test_number), *paths],
                              cwd=root, env=env, start_new_session=True, stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as child:
            try:
                return True if child.wait(timeout=timeout) == 83 else None
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
    except (OSError, ValueError, TypeError, subprocess.TimeoutExpired, EOFError):
        return None


def check(number, paths):
    from proxy_guard import Guard
    from proxy_phase import encode, replay

    guard = Guard()
    root = Path.cwd()
    for path in paths:
        guard.add(path, root)
    sys.meta_path.insert(0, guard)
    with contextlib.ExitStack() as stack:
        streams = {}
        for file in MAPS.glob(f"{number}.*.gz"):
            kind = file.name.split(".")[1]
            if kind not in ("guard", "inputs"):
                streams[kind] = stack.enter_context(gzip.open(file, "rb"))

        def emit(group, value):
            if streams[group].readline() != encode(value) + b"\n":
                raise ValueError(f"boundary mismatch in {group}")

        inputs = json.loads(gzip.decompress((MAPS / f"{number}.inputs.gz").read_bytes()))
        replay(number, inputs, emit)
        if not guard.clean or not streams or any(f.read(1) for f in streams.values()):
            raise ValueError("incomplete boundary stream")


if __name__ == "__main__":
    check(int(sys.argv[1]), sys.argv[2:])
    sys.exit(83)
