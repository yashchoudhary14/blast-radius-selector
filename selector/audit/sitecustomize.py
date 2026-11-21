"""Used only by the offline source-read map builder, including spawned interpreters."""

import os
import sys


class MissingModules:
    """Last finder in the baseline interpreter. A new source file could satisfy this lookup."""

    def __init__(self, record):
        self.record = record

    def find_spec(self, fullname, path, target=None):
        if path is None or any(p.startswith("/sqlglot/") for p in path):
            self.record(fullname.replace(".", "/") + ".py")
        return None


def install():
    sink = os.environ.get("BR_AUDIT_LOG")
    if not sink:
        return
    fd = os.open(sink, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    seen = set()

    def audit(event, args):
        if event in {"os.listdir", "os.scandir"}:
            path = args[0]
            if not isinstance(path, (str, bytes)):
                return
            path = os.path.abspath(os.fsdecode(path))
            if path != "/sqlglot" and not path.startswith("/sqlglot/"):
                return
            if sys._getframe(1).f_code.co_filename == "<frozen importlib._bootstrap_external>":
                return
            record = (os.environ.get("BR_AUDIT_TEST", ""), "directory",
                      os.path.relpath(path, "/sqlglot"))
            if record not in seen:
                seen.add(record)
                os.write(fd, ("\t".join(record) + "\n").encode())
            return
        if event != "open":
            return
        path, _, flags = args
        if not isinstance(path, str) or not path.endswith(".py") or flags & os.O_ACCMODE == os.O_WRONLY:
            return
        path = os.path.abspath(path)
        if not path.startswith("/sqlglot/"):
            return
        frame = sys._getframe(1)
        loader = frame.f_code.co_filename == "<frozen importlib._bootstrap_external>" and (
            frame.f_code.co_name == "get_data")
        context = os.environ.get("BR_AUDIT_TEST", "")
        record = (context, "import" if loader else "read", path.removeprefix("/sqlglot/"))
        if record not in seen:
            seen.add(record)
            os.write(fd, ("\t".join(record) + "\n").encode())

    sys.addaudithook(audit)


install()


def record_child_imports():
    sink = os.environ.get("BR_CHILD_IMPORT_LOG")
    if not sink:
        return
    fd = os.open(sink, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    seen = set()

    def missing(path):
        if path not in seen:
            seen.add(path)
            os.write(fd, (path + "\n").encode())

    sys.meta_path.append(MissingModules(missing))

    def audit(event, args):
        if event != "exec":
            return
        code = args[0]
        path = code.co_filename
        if code.co_name == "<module>" and path.startswith("/sqlglot/") and path not in seen:
            seen.add(path)
            os.write(fd, (path.removeprefix("/sqlglot/") + "\n").encode())

    sys.addaudithook(audit)


record_child_imports()
