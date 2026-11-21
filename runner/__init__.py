"""Run selected sqlglot unittest methods in an isolated Linux process."""

import importlib
import json
import os
import resource
import select
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .cases import GENERATOR_MEMORY, execute
from .overlay import PatchError, changes, populate
from .process import child_pipe
from .query_cases import QUERIES
from .warm import WarmParents, project_modules


@dataclass
class RunResult:
    failing: set[str] = field(default_factory=set)
    ran: list[str] = field(default_factory=list)
    cpu_s: float = 0.0
    import_error: str | None = None
    notes: list[str] = field(default_factory=list)


def _usage():
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def cpu_time():
    # wait4 only charges a snapshot's children to the lane when it exits.
    return time.process_time() + _usage() + sum(parent.cpu for parent in WarmParents.live)


def _enter_tree(root, base, warmed, *, pin=True):
    os.chdir(root)
    sys.path[:] = [str(root)] + [
        p for p in sys.path if p and os.path.abspath(p) not in (str(root), str(base))
    ]
    os.environ["PYTHONPATH"] = os.pathsep.join(sys.path)
    os.environ["SKIP_INTEGRATION"] = "0"
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    if not warmed:
        for name in project_modules():
            del sys.modules[name]
    importlib.invalidate_caches()

    # Docker's CPU quota does not constrain sched_getaffinity(). Python
    # otherwise spawns one heavy pandas/duckdb process per host CPU.
    cpus = sorted(os.sched_getaffinity(0))
    if pin and len(cpus) > 2:
        current = int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[36])
        start = cpus.index(current) if current in cpus else 0
        os.sched_setaffinity(0, [cpus[start], cpus[(start + 1) % len(cpus)]])
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    # Spawned test workers import functions from their test module. Replaying
    # the selector's top-level program can print predictions or start forks.
    main = sys.modules["__main__"]
    main.__dict__.pop("__file__", None)
    main.__spec__ = None


def _child(fd, root, base, ids, deadlines, early, warmed, case_filter):
    def send(*event):
        data = json.dumps(event, separators=(",", ":")).encode() + b"\n"
        while data:
            data = data[os.write(fd, data):]

    try:
        with open(os.devnull, "wb") as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
        _enter_tree(root, base, warmed)
        execute(ids, deadlines, early, send, case_filter)
        send("done")
        os._exit(0)
    except BaseException as exc:
        send("note", f"runner child: {type(exc).__name__}: {exc}")
        os._exit(1)


def _collect(fd, ids, deadline):
    result = RunResult()
    seen = set()
    active = []
    done = False
    pending = b""
    while not done:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
            result.notes.append("overall timeout")
            break
        chunk = os.read(fd, 65536)
        if not chunk:
            break
        *events, pending = (pending + chunk).split(b"\n")
        for line in events:
            kind, *args = json.loads(line)
            if kind == "start":
                active = [args[0]]
                if args[0] not in seen:
                    seen.add(args[0])
                    result.ran.append(ids[args[0]])
            elif kind == "end":
                active = []
            elif kind == "fail":
                result.failing.add(ids[args[0]])
            elif kind == "group":
                for idx in args[0]:
                    if idx not in seen:
                        seen.add(idx)
                        result.ran.append(ids[idx])
                    if args[1]:
                        result.failing.add(ids[idx])
                active = []
            elif kind == "scope":
                active = args[0]
            elif kind == "import":
                result.import_error = result.import_error or args[0]
                result.notes.append(args[0])
            elif kind == "note":
                result.notes.append(args[0])
            elif kind == "done":
                done = True
    if not done:
        for idx in active:
            if idx not in seen:
                result.ran.append(ids[idx])
            result.failing.add(ids[idx])
        result.notes.append("incomplete child run")
    return result


def _fork_run(root, base, ids, deadlines, early, deadline, case_filter, *, warm):
    start = _usage()
    with child_pipe(_child, root, base, ids, deadlines, early, warm, case_filter) as rd:
        result = _collect(rd, ids, deadline)
    result.cpu_s = _usage() - start
    return result


def _state_child(fd, root, base, warm):
    from .state import diff, ImportFailure
    try:
        with open(os.devnull, "wb") as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
        _enter_tree(root, base, warm)
        changes = diff(root)
        result = ("ok", [(owner, key, *info) for (owner, key), info in changes.items()])
    except ImportFailure:
        result = ("broken", [])
    except BaseException:
        result = ("unknown", [])
    with os.fdopen(fd, "wb") as output:
        output.write(json.dumps(result).encode())
    os._exit(0)


def _fork_state(root, base, deadline, *, warm):
    start = _usage()
    if warm:
        from .state import preload
        preload()
    data = bytearray()
    with child_pipe(_state_child, root, base, warm) as fd:
        while len(data) < 1048576:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                break
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            data.extend(chunk)
    try:
        status, rows = json.loads(data)
    except ValueError:
        status, rows = "unknown", []
    return status, rows, _usage() - start


class Session:
    def __init__(self, base="/sqlglot", tmp="/tmp/br", *, warm_fixtures=True):
        self.base = Path(base).resolve()
        self.tmp = Path(tmp).resolve()
        self.tmp.mkdir(parents=True, exist_ok=True)
        self._temp = tempfile.TemporaryDirectory(dir=self.tmp)
        self._root = Path(self._temp.name) / "tree"
        self.warm_fixtures = warm_fixtures
        self._warm = None
        self.seconds = {}
        catalog = Path("/data/tests.json")
        if catalog.is_file():
            self.seconds = {t["id"]: t["seconds"] for t in json.loads(catalog.read_bytes())["tests"]}

    def warm(self):
        if self.warm_fixtures and self._warm is None and self.base == Path("/sqlglot"):
            self._clear()
            populate(self.base, self._root, {})
            try:
                self._warm = WarmParents(self._root, self.base, self.seconds, _enter_tree,
                                         {"tests": _fork_run, "state": _fork_state})
                print(f"lane_preload_cpu={self._warm.preload_cpu:.6f}", file=sys.stderr, flush=True)
            except (OSError, EOFError) as exc:
                print(f"runner preload failed: {exc!r}", file=sys.stderr)
                self.warm_fixtures = False
            finally:
                self._clear()
        return self._warm.preload_cpu if self._warm else 0.

    def import_stage(self, paths):
        self.warm()
        return self._warm.stage(paths) if self._warm else -1

    def import_state(self, edits, deadline):
        """Only stage zero is production-only. Fall back to a fresh child for core edits."""
        self._clear()
        try:
            populate(self.base, self._root, edits)
            if self._warm and self._warm.stage(edits) >= 0:
                parent = self._warm.snapshots[0][0]
                parent.send(("state", (deadline,)))
                status, rows, cpu = parent.recv()
                self._warm.cpu += cpu
            else:
                status, rows, cpu = _fork_state(self._root, self.base, deadline, warm=False)
            return status, rows, cpu
        finally:
            self._clear()

    def close(self):
        if self._warm:
            self._warm.close()
            self._warm = None
        self._temp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _clear(self):
        if self._root.exists():
            shutil.rmtree(self._root)

    def apply_only(self, patch_text):
        self._clear()
        try:
            edits = changes(self.base, patch_text)
            populate(self.base, self._root, edits)
            return self._root
        except BaseException:
            self._clear()
            raise

    def run(self, patch_text, test_ids, *, timeout_factor=10.0, stop_on_first_fail=False,
            per_test_budget=None, overall_timeout=None, case_filter=None):
        self._clear()
        try:
            edits = changes(self.base, patch_text)
            ids = list(dict.fromkeys(test_ids))
            if case_filter and set(case_filter) - ((QUERIES.keys() | {GENERATOR_MEMORY}) & set(ids)):
                raise ValueError("case filter requires a supported, selected test")
            if not ids:
                return RunResult()
            self.warm()
            parent = self._warm.choose(edits, self.base, ids, patch_text) if self._warm else None
            populate(self.base, self._root, edits)
            budgets = per_test_budget or {}
            deadlines = [max(1.0, timeout_factor * budgets.get(i, self.seconds.get(i, 0))) for i in ids]
            guard = overall_timeout if overall_timeout is not None else 15 + sum(deadlines)
            deadline = time.monotonic() + guard
            request = (ids, deadlines, stop_on_first_fail, deadline, case_filter)
            if parent is not None:
                try:
                    parent.send(("tests", request))
                    result = parent.recv()
                except (EOFError, OSError) as exc:
                    print(f"runner snapshot unavailable: {exc!r}", file=sys.stderr)
                    self._warm.close()
                    self._warm = None
                    self.warm_fixtures = False
                else:
                    self._warm.cpu += result.cpu_s
                    return result
            return _fork_run(self._root, self.base, *request, warm=False)
        finally:
            self._clear()
