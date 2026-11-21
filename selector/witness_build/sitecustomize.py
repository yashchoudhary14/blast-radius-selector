"""Collect entry facts in pristine test interpreters, including their spawned workers."""
import atexit
import functools
import json
import os
import sys
import types
from pathlib import Path

ROOT = "/sqlglot/sqlglot/"
NON_NULL = 1
TRUE = 2
FALSE = 4
facts = {}
modules = set()
inspected = set()
lines = {}
reads = set()
tool = 4


@functools.lru_cache(maxsize=1024)
def ordinary_truth(cls):
    # Only use the default object truth operation for normal SQLGlot objects.
    return (cls.__module__.startswith("sqlglot.") and
            all("__bool__" not in t.__dict__ and "__len__" not in t.__dict__ for t in cls.__mro__))


def value_mask(value):
    cls = type(value)
    if cls in (dict, tuple, list, str, int, bool, set, bytes, frozenset, float, type(None)):
        truth = TRUE if value else FALSE
    elif ordinary_truth(cls):
        truth = TRUE
    else:
        truth = 0
    return truth | (NON_NULL if value is not None else 0)


def entered(code, offset):
    if not code.co_filename.startswith(ROOT):
        return sys.monitoring.DISABLE
    if code.co_name == "<module>":
        modules.add(code.co_filename)
        return sys.monitoring.DISABLE
    if code.co_name.startswith("<") or code.co_flags & (0x20 | 0x80 | 0x200):
        return sys.monitoring.DISABLE
    params = facts.get(code)
    if params is None:
        params = facts[code] = dict.fromkeys(code.co_varnames[:code.co_argcount + code.co_kwonlyargcount], 7)
    loc = sys._getframe(1).f_locals
    for name in params:
        if params[name]:
            params[name] &= value_mask(loc[name]) if name in loc else 0
    return None


def line(code, lineno):
    if code.co_filename.startswith(ROOT):
        lines.setdefault(code.co_filename, set()).add(lineno)
    return sys.monitoring.DISABLE


def audit(event, args):
    if event == "open" and isinstance(args[0], str) and args[0].startswith(ROOT) and args[0].endswith(".py"):
        frame = sys._getframe(1)
        if (frame.f_code.co_filename == "<frozen importlib._bootstrap_external>" and frame.f_code.co_name == "get_data"):
            return
        # The instrumenting loader reads the original .py for compilation.
        caller = frame
        while caller and caller.f_globals.get("__name__") in ("pathlib", "pathlib._local", "pathlib._abc"):
            caller = caller.f_back
        if not (caller and caller.f_globals.get("__name__") == "identity" and caller.f_code.co_name == "get_code"):
            reads.add(args[0])

    if event in ("object.__getattr__", "object.__setattr__") and type(args[0]) is types.FunctionType:
        if args[1] in ("__defaults__", "__kwdefaults__", "__code__"):
            frame = sys._getframe(1)
            # inspect.signature uses argument slots, which a guard before the
            # first statement cannot change. It can observe a revised default.
            signature = (frame.f_globals.get("__name__") == "inspect" and
                         frame.f_code.co_name == "_signature_from_function")
            overload = (frame.f_globals.get("__name__") == "typing" and frame.f_code.co_name == "overload")
            # CPython 3.13 typing.overload reads only co_firstlineno on the decorated stub.
            if overload:
                return
            mode = "signature" if signature else "metadata"
            inspected.add((args[0].__module__, args[0].__qualname__, mode))


def install():
    from identity import Finder, OUTCOMES
    sys.meta_path.insert(0, Finder())
    out = Path(os.environ["WITNESS_DIR"]) / str(os.getpid())
    out.touch()

    def save():
        sys.monitoring.set_events(tool, 0)
        rows = {}
        for code, params in facts.items():
            key = f"{code.co_filename.removeprefix('/sqlglot/')}:{code.co_firstlineno}"
            prior = rows.setdefault(key, params.copy())
            for p in list(prior):
                prior[p] &= params.get(p, 0)
        out.write_text(json.dumps([sorted(modules), rows, sorted(inspected, key=str), {k: sorted(v) for k,v in lines.items()}, sorted(reads), OUTCOMES]))

    atexit.register(save)
    sys.addaudithook(audit)
    sys.monitoring.use_tool_id(tool, "selector pristine calls")
    sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, entered)
    sys.monitoring.register_callback(tool, sys.monitoring.events.LINE, line)
    sys.monitoring.set_events(tool, sys.monitoring.events.PY_START | sys.monitoring.events.LINE)


if os.environ.get("WITNESS_DIR"):
    install()
