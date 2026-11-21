"""Run a Linux test process group and reap helpers orphaned by a test or timeout."""

import ctypes
import os
import signal
from contextlib import contextmanager


_prctl = ctypes.CDLL(None, use_errno=True).prctl
PR_SET_CHILD_SUBREAPER = 36
PR_GET_CHILD_SUBREAPER = 37


def _subreaper(option, arg):
    if _prctl(option, arg, 0, 0, 0) == -1:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


@contextmanager
def child_pipe(target, *args):
    # A selector worker may be several forks below container PID 1. Adopt
    # abandoned grandchildren here so waitpid accounts their CPU and frees PIDs.
    previous = ctypes.c_int()
    _subreaper(PR_GET_CHILD_SUBREAPER, ctypes.byref(previous))
    enable = os.getpid() != 1 and not previous.value
    if enable:
        _subreaper(PR_SET_CHILD_SUBREAPER, 1)
    try:
        rd, wr = os.pipe()
        try:
            pid = os.fork()
        except BaseException:
            os.close(rd)
            os.close(wr)
            raise
        if pid == 0:
            try:
                os.close(rd)
                os.setpgid(0, 0)
                target(wr, *args)
            finally:
                os._exit(1)
        os.close(wr)
        try:
            os.setpgid(pid, pid)
            yield rd
        finally:
            os.close(rd)
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            os.waitpid(pid, 0)
            # Only reap this group's orphans. The caller can own other children.
            while True:
                try:
                    os.waitpid(-pid, 0)
                except ChildProcessError:
                    break
    finally:
        if enable:
            _subreaper(PR_SET_CHILD_SUBREAPER, 0)
