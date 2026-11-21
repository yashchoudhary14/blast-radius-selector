"""Install observers in spawned Python workers and require their exit records."""
import atexit
import os
import sys
from pathlib import Path


if os.environ.get("BR_OPERAND_BUILD"):
    import marshal
    import record

    record.install()
    directory = os.environ.get("BR_OPERAND_TEST")
    if directory and not any("multiprocessing.resource_tracker import main" in arg for arg in sys.orig_argv):
        target = Path(directory) / str(os.getpid())
        target.touch()

        @atexit.register
        def save():
            target.write_bytes(marshal.dumps((dict(record.vectors), record.bounds)))
