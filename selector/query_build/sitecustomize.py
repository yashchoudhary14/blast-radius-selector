"""Record every process of one isolated pristine case, including pool workers."""

import atexit
import json
import os
import sys
from pathlib import Path


def install(directory):
    # Resource trackers execute no project code and may outlive the parent.
    if any("multiprocessing.resource_tracker import main" in arg for arg in sys.orig_argv):
        return
    out = Path(directory) / str(os.getpid())
    out.touch()
    lines = {}
    tool = 4

    def line(code, number):
        path = code.co_filename
        if path.startswith(("/sqlglot/sqlglot/", "/sqlglot/tests/")):
            lines.setdefault(path[9:], set()).add(number)
        return sys.monitoring.DISABLE

    def save():
        sys.monitoring.set_events(tool, 0)
        out.write_text(json.dumps({p: sorted(v) for p, v in lines.items()}))

    sys.monitoring.use_tool_id(tool, "fixture case coverage")
    sys.monitoring.register_callback(tool, sys.monitoring.events.LINE, line)
    sys.monitoring.set_events(tool, sys.monitoring.events.LINE)
    atexit.register(save)


if os.environ.get("BR_QUERY_CONTEXT") and not any(
        "multiprocessing.resource_tracker import main" in arg for arg in sys.orig_argv):
    from contexts import install as install_contexts
    install_contexts(os.environ["BR_QUERY_CONTEXT"])
elif os.environ.get("BR_QUERY_TRACE"):
    install(os.environ["BR_QUERY_TRACE"])
