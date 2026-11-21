import py_compile
import subprocess
import sys
import tempfile
import textwrap
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from . import GENERATOR_MEMORY, PatchError, Session


class RunnerTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name) / "base"
        (self.base / "tests").mkdir(parents=True)
        self.put("tests/__init__.py", "")
        self.session = Session(self.base, Path(tmp.name) / "overlays")
        self.addCleanup(self.session.close)

    def put(self, path, code):
        (self.base / path).write_text(textwrap.dedent(code).lstrip("\n"))

    def test_overlay(self):
        (self.base / "a").write_bytes(b"first\r\nsecond\r\nlast")
        self.put("gone", "bye")
        diff = """--- a/a
+++ b/a
@@ -1,2 +1,2 @@
 first
-second
+two
@@ -3 +3 @@
-last
\\ No newline at end of file
+end
\\ No newline at end of file
--- /dev/null
+++ b/new/nested
@@ -0,0 +1 @@
+created
--- a/gone
+++ /dev/null
@@ -1 +0,0 @@
-bye
\\ No newline at end of file
"""
        root = self.session.apply_only(diff)
        self.assertEqual((root / "a").read_bytes(), b"first\r\ntwo\r\nend")
        self.assertEqual((root / "new/nested").read_bytes(), b"created\n")
        self.assertFalse((root / "gone").exists())
        self.assertTrue((root / "tests").is_symlink())
        with self.assertRaises(PatchError):
            self.session.apply_only(diff.replace("-second", "-absent"))
        self.assertFalse(root.exists())

    def test_caches_and_subprocess(self):
        self.put("tests/mod.py", "X = 1\n")
        py_compile.compile(str(self.base / "tests/mod.py"), doraise=True,
                           invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)
        self.put("tests/test_mod.py", """
            import subprocess, sys, unittest
            from tests.mod import X
            class T(unittest.TestCase):
                def test_val(self):
                    self.assertEqual(X, 2)
                    subprocess.run([sys.executable, "-c", "from tests.mod import X; assert X == 2"],
                                   check=True)
        """)
        ids = ["tests/test_mod.py::T::test_val"]
        diff = "--- a/tests/mod.py\n+++ b/tests/mod.py\n@@ -1 +1 @@\n-X = 1\n+X = 2\n"
        inherited = types.ModuleType("tests.mod")
        inherited.X = 99
        with patch.dict(sys.modules, {"tests.mod": inherited}):
            self.assertFalse(self.session.run(diff, ids).failing)
            self.assertEqual(self.session.run("", ids).failing, set(ids))
            self.assertIs(sys.modules["tests.mod"], inherited)

    def test_fixture_and_method_outcomes(self):
        self.put("tests/bad.py", "raise ImportError('missing')\n")
        self.put("tests/cases.py", """
            import unittest
            class A(unittest.TestCase):
                @classmethod
                def setUpClass(cls):
                    raise ValueError()
                def test_x(self): pass
                def test_y(self): pass
            class B(unittest.TestCase):
                def test_sub(self):
                    with self.subTest():
                        self.fail()
                @unittest.expectedFailure
                def test_unexpected(self): pass
                @unittest.skip("skip")
                def test_skip(self): pass
                @unittest.expectedFailure
                def test_expected(self): self.fail()
        """)
        ids = ["tests/cases.py::" + i for i in (
            "A::test_x", "A::test_y", "B::test_sub", "B::test_unexpected",
            "B::test_skip", "B::test_expected")]
        ids += ["tests/bad.py::C::test_x", "tests/bad.py::C::test_y"]
        r = self.session.run("", ids)
        self.assertEqual(set(r.ran), set(ids))
        self.assertEqual(r.failing, set(ids[:4] + ids[6:]))
        self.assertIn("ImportError", r.import_error)
        r = self.session.run("", ids[2:], stop_on_first_fail=True)
        self.assertEqual(r.ran, ids[2:3])
        self.assertEqual(r.failing, set(r.ran))
        self.assertIsNone(r.import_error)

    def test_dialect_case_filter(self):
        self.put("tests/test_generator.py", """
            import enum, unittest
            class Dialects(enum.Enum):
                A, B, C = range(3)
            original = Dialects
            class TestGenerator(unittest.TestCase):
                def test_generator_memory(self):
                    for member in Dialects:
                        with self.subTest(member=member):
                            self.assertNotEqual(member, original.B)
                    if original.B not in Dialects:
                        self.assertEqual(list(Dialects), [original.A, original.C])
                def tearDown(self):
                    self.assertIs(Dialects, original)
                def test_restored(self):
                    self.assertIs(Dialects, original)
        """)
        ids = [GENERATOR_MEMORY, GENERATOR_MEMORY.replace("test_generator_memory", "test_restored")]
        for names, failures in [(["C", "A"], set()), (["B"], {ids[0]})]:
            result = self.session.run("", ids, case_filter={ids[0]: names})
            self.assertEqual(result.ran, ids)
            self.assertEqual(result.failing, failures)
            self.assertEqual(result.notes, [])

    def test_module_setup_and_teardown(self):
        for phase in ("setUpModule", "tearDownModule", "tearDownClass"):
            body = "class T(unittest.TestCase):\n    def test_x(self): pass\n    def test_y(self): pass\n"
            if phase.endswith("Class"):
                body += f"    @classmethod\n    def {phase}(cls): raise ValueError()\n"
            else:
                body += f"def {phase}(): raise ValueError()\n"
            self.put("tests/cases.py", "import unittest\n" + body)
            ids = ["tests/cases.py::T::test_" + x for x in "xy"]
            r = self.session.run("", ids)
            self.assertEqual(r.ran, ids)
            self.assertEqual(r.failing, set(ids if phase.startswith("setUp") else ids[-1:]))

    def test_early_teardown_failure(self):
        self.put("tests/cases.py", """
            import unittest
            class A(unittest.TestCase):
                @classmethod
                def tearDownClass(cls): raise ValueError()
                def test_x(self): pass
            class B(unittest.TestCase):
                def test_x(self): self.fail()
        """)
        ids = [f"tests/cases.py::{cls}::test_x" for cls in ("A", "B")]
        result = self.session.run("", ids, stop_on_first_fail=True)
        self.assertEqual(result.ran, ids[:1])
        self.assertEqual(result.failing, set(ids[:1]))

    def test_shared_failure_scope(self):
        self.put("tests/ok.py", """
            import unittest
            class T(unittest.TestCase):
                def test_ok(self): pass
        """)
        ids = ["tests/bad.py::T::test_x", "tests/bad.py::T::test_y", "tests/ok.py::T::test_ok"]
        for source in ("raise ImportError('broken')", """
            import unittest
            def setUpModule(): raise ValueError()
            class T(unittest.TestCase):
                def test_x(self): pass
                def test_y(self): pass
        """):
            self.put("tests/bad.py", source)
            for early in (True, False):
                with self.subTest(source=source, early=early):
                    result = self.session.run("", ids, stop_on_first_fail=early)
                    self.assertEqual(result.failing, set(ids[:2]))
                    self.assertEqual(result.ran, ids[:2] if early else ids)
                    self.assertEqual(bool(result.import_error), source.startswith("raise ImportError"))

    def test_literal_cr_and_unsafe_paths(self):
        (self.base / "a").write_bytes(b"one\n")
        patch = "--- a/a\n+++ b/a\n@@ -1 +1 @@\n-one\n+one\r\n"
        root = self.session.apply_only(patch)
        self.assertEqual((root / "a").read_bytes(), b"one\r\n")
        with self.assertRaises(PatchError):
            self.session.apply_only(patch.replace("+++ b/a", "+++ b/../escaped"))

    @unittest.skipUnless(Path("/sqlglot/sqlglotc/setup.py").is_file(), "needs pinned base")
    def test_resolved_source_path(self):
        file = "sqlglotc/setup.py"
        patch = f"--- a/{file}\n+++ b/{file}\n@@ -0,0 +1 @@\n+raise ValueError('setup')\n"
        ids = ["tests/test_sqlglotc.py::TestSQLGlotC::test_build_removes_obsolete_extensions"]
        with Session() as session:
            self.assertEqual(session.run(patch, ids).failing, set(ids))

    @unittest.skipUnless(Path("/sqlglot/tests/fixtures/identity.sql").is_file(), "needs pinned base")
    def test_fixture_code_fixture_switch(self):
        file = "tests/fixtures/identity.sql"
        fixture = f"--- a/{file}\n+++ b/{file}\n@@ -0,0 +1 @@\n+select 1\n"
        file = "sqlglot/parser.py"
        broken = f"--- a/{file}\n+++ b/{file}\n@@ -0,0 +1 @@\n+invalid python!\n"
        ids = ["tests/test_transpile.py::TestTranspile::test_identity"]
        original = sys.modules.get("sqlglot")
        with Session() as session:
            self.assertEqual(session.run(fixture, ids).failing, set(ids))
            error = session.run(broken, ids)
            self.assertEqual(error.failing, set(ids))
            self.assertIn("SyntaxError", error.import_error)
            self.assertFalse(session.run(fixture.replace("select 1", "SELECT 1"), ids).failing)
        self.assertIs(sys.modules.get("sqlglot"), original)

    def test_deadlines(self):
        self.put("tests/cases.py", """
            import unittest, time
            class T(unittest.TestCase):
                def test_slow(self):
                    with self.subTest():
                        time.sleep(5)
                    time.sleep(5)
                def test_ok(self): pass
        """)
        ids = ["tests/cases.py::T::test_slow", "tests/cases.py::T::test_ok"]
        start = time.monotonic()
        r = self.session.run("", ids, timeout_factor=1, per_test_budget={ids[0]: 1.1})
        self.assertEqual(r.ran, ids)
        self.assertEqual(r.failing, {ids[0]})
        self.assertLess(time.monotonic() - start, 3)
        r = self.session.run("", ids, overall_timeout=0.2)
        self.assertEqual(r.failing, {ids[0]})
        self.assertEqual(r.ran, ids[:1])
        self.assertIn("overall timeout", r.notes)

    def test_orphan_cpu_and_reaping(self):
        marker = self.base.parent / "worker.pid"
        worker = ("import os, time; from pathlib import Path\n"
                  "while time.process_time() < 0.17: pass\n"
                  f"Path({str(marker)!r}).write_text(str(os.getpid()))\n"
                  "time.sleep(30)\n")
        self.put("tests/spawn.py", f"""
            import subprocess, sys, time, unittest
            from pathlib import Path
            class T(unittest.TestCase):
                def test_spawn(self):
                    subprocess.Popen([sys.executable, "-c", {worker!r}])
                    while not Path({str(marker)!r}).exists(): time.sleep(0.01)
        """)
        # Check a non-PID-1 selector worker: an ordinary killpg leaves the
        # orphan with container init and omits the worker's CPU from r.cpu_s.
        script = f"""
            from pathlib import Path
            from runner import Session
            with Session({str(self.base)!r}, {str(self.session.tmp)!r}) as s:
                node = 'tests/spawn.py::T::test_spawn'
                r = s.run('', [node], per_test_budget={{node: 1}})
                assert not r.failing and r.ran == [node], r
                assert r.cpu_s >= 0.15, r.cpu_s
                pid = Path({str(marker)!r}).read_text()
                assert not Path('/proc', pid).exists(), pid
        """
        subprocess.run([sys.executable, "-c", textwrap.dedent(script)], check=True, timeout=20)


if __name__ == "__main__":
    unittest.main()
