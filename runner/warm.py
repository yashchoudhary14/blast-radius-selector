"""Fork snapshots at import boundaries in a private process tree.

A snapshot predating the first import of every changed file also predates its
importers and all its registry/cache writes, including writes to older modules.
Later stages live in descendants, so none of their mutations can travel back.
"""
import importlib
import json
import os
import re
import sys
import time
from multiprocessing import Pipe
from pathlib import Path


def project_modules():
    return {name: module for name, module in sys.modules.copy().items()
            if name.partition(".")[0] in ("tests", "sqlglot", "sqlglotc", "benchmarks")}


# Changes observing interpreter state must see exactly the cold import order.
IMPORT_STATE = re.compile(r"^[+-].*(?:sys\.modules|__subclasses__|_classes\b|gc\.|tracemalloc|"
                          r"importlib\.|__import__|sys\.meta_path)", re.MULTILINE)
PLUGIN_TEST = "tests/test_dialect_entry_points.py"


def stages(base, tests):
    importlib.import_module("duckdb")
    importlib.import_module("pandas.testing")
    importlib.import_module("sqlglot")
    yield

    for name in ("tests.test_optimizer", "tests.test_executor", "tests.dialects.test_dialect"):
        importlib.import_module(name)
    yield

    for path in sorted((base / "sqlglot").rglob("*.py")):
        # The CLI consumes argv/stdin and exits at import time.
        if path.name == "__main__.py":
            continue
        parts = path.relative_to(base).with_suffix("").parts
        name = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        importlib.import_module(name)
    files = {node.partition("::")[0] for node in tests}
    if not files:
        files = {str(path.relative_to(base)) for path in (base / "tests").rglob("test_*.py")}
    for file in sorted(files - {PLUGIN_TEST}):
        importlib.import_module(file[:-3].replace("/", "."))
    yield
    # This test defines FakeDialect at module scope. Its metaclass registers
    # that name globally; only that test file may inherit this last snapshot.
    importlib.import_module(PLUGIN_TEST[:-3].replace("/", "."))
    yield


def serve_stages(channels, root, base, tests, enter, operations):
    # Test children choose affinity at fork just like cold children. Pinning
    # this long-lived ancestor would put every later probe on its startup CPUs.
    enter(root, base, False, pin=False)
    started = time.process_time()
    child = None
    try:
        for index, _ in enumerate(stages(base, tests)):
            files = {str(Path(mod.__file__).relative_to(root))
                     for mod in project_modules().values() if getattr(mod, "__file__", None)}
            cpu = time.process_time() - started
            conn = channels[index]
            conn.send((files, cpu))
            if index + 1 < len(channels):
                child = os.fork()
                if not child:
                    conn.close()
                    child = None
                    started = time.process_time()
                    continue
            for other in channels[index + 1:]:
                other.close()
            while (request := conn.recv()) is not None:
                operation, args = request
                conn.send(operations[operation](root, base, *args, warm=True))
            break
    finally:
        for conn in channels:
            conn.close()
        if child:
            os.waitpid(child, 0)


class WarmParents:
    live = set()

    def __init__(self, root, base, tests, enter, operations):
        graph = Path("/app/maps/runner_importers.json")
        self.importers = json.loads(graph.read_bytes()) if graph.is_file() else {}
        pairs = [Pipe() for _ in range(4)]
        self.pid = os.fork()
        if not self.pid:
            for client, _ in pairs:
                client.close()
            try:
                serve_stages([server for _, server in pairs], root, base, tests, enter, operations)
            except (EOFError, BrokenPipeError):
                pass
            except BaseException:
                import traceback
                traceback.print_exc(file=sys.stderr)
            finally:
                os._exit(0)
        self.clients = []
        self.snapshots = []
        self.cpu = 0.
        for client, server in pairs:
            server.close()
            self.clients.append(client)
        try:
            for client in self.clients:
                files, cpu = client.recv()
                self.snapshots.append((client, files))
                self.cpu += cpu
            self.preload_cpu = self.cpu
            self.live.add(self)
        except BaseException:
            self.close()
            raise

    def close(self):
        self.live.discard(self)
        for client in self.clients:
            try:
                client.send(None)
            except (BrokenPipeError, OSError):
                pass
            client.close()
        os.waitpid(self.pid, 0)

    def stage(self, paths, include_plugin=False):
        affected = set(paths)
        pending = list(affected)
        while pending:
            for importer in self.importers.get(pending.pop(), ()):
                if importer not in affected:
                    affected.add(importer)
                    pending.append(importer)
        count = len(self.snapshots) if include_plugin else len(self.snapshots) - 1
        for index in reversed(range(count)):
            if self.snapshots[index][1].isdisjoint(affected):
                return index
        return -1

    def choose(self, edits, base, ids, patch):
        for path, data in edits.items():
            if path.startswith("tests/fixtures/") and path.endswith((".sql", ".json", ".gz")):
                # The pinned collection-read audit found no fixture content
                # reads on import. Test and class setup remain child-local.
                continue
            # New/deleted Python paths can change package discovery.
            if data is None or not (base / path).is_file():
                return None
            # Patched tests may directly inspect import order, registries or
            # the live heap. Run their new assertions in the reference process.
            if not path.startswith("sqlglot/") or not path.endswith(".py"):
                return None
        if any(path.endswith(".py") for path in edits) and IMPORT_STATE.search(patch):
            return None
        stage = self.stage(edits, all(node.partition("::")[0] == PLUGIN_TEST for node in ids))
        return self.snapshots[stage][0] if stage >= 0 else None
