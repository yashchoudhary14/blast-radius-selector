"""A private, copy-on-fork cache of pristine Python modules for SQL fixture edits."""

import importlib
import logging
import os
import sys
from unittest import mock


def project_modules():
    return {
        name: module for name, module in sys.modules.copy().items()
        if name.split(".", 1)[0] in ("tests", "sqlglot", "sqlglotc", "benchmarks")
    }


def preload(root, modules=("tests.test_parser",)):
    # These pinned modules load no fixture values during import. Helpers derive
    # FIXTURES_DIR from __file__, which therefore already names the overlay.
    # Keep native DB connections and all test execution in the forked child.
    original = project_modules()
    old_path = sys.path[:]
    old_flag = sys.dont_write_bytecode
    try:
        for name in original:
            del sys.modules[name]
        sys.path.insert(0, str(root))
        sys.dont_write_bytecode = True
        importlib.invalidate_caches()
        with (mock.patch.dict(os.environ, SKIP_INTEGRATION="0"),
              mock.patch.object(logging.getLogger("sqlglot"), "disabled", True)):
            for name in modules:
                importlib.import_module(name)
        return project_modules()
    finally:
        for name in project_modules():
            del sys.modules[name]
        sys.modules.update(original)
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_flag
