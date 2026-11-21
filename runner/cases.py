"""Translate unittest lifecycle events into the scorer's method-level labels."""

import importlib
import signal
import unittest
from collections import defaultdict
from functools import wraps
from itertools import groupby

from .parser_cases import MAX_NODES, filter_parser
from .query_cases import filter_queries


GENERATOR_MEMORY = "tests/test_generator.py::TestGenerator::test_generator_memory"


def filter_dialects(case, module, names):
    method = getattr(case, case._testMethodName)
    wanted = set(names)

    @wraps(method)
    def filtered():
        original = module.Dialects
        members = [member for member in original if member.name in wanted]
        if len(members) != len(wanted):
            raise ValueError("unknown Dialects member in case filter")
        try:
            module.Dialects = members
            return method()
        finally:
            module.Dialects = original

    setattr(case, case._testMethodName, filtered)


class TimedOut(BaseException):
    pass


class Result(unittest.TestResult):
    def __init__(self, send, deadlines, early):
        super().__init__()
        self.send = send
        self.deadlines = deadlines
        self.failfast = early
        self.by_scope = defaultdict(list)
        self.active = None
        self.last = None
        self.expired = False
        signal.signal(signal.SIGALRM, self._alarm)

    def _alarm(self, signum, frame):
        if self.active is None:
            raise TimeoutError("shared fixture or import timed out")
        self.expired = True
        raise TimedOut()

    def startTest(self, test):
        super().startTest(test)
        self.active = test._runner_index
        self.last = self.active
        self.send("start", self.active)
        self.expired = False
        signal.setitimer(signal.ITIMER_REAL, self.deadlines[self.active])

    def stopTest(self, test):
        signal.setitimer(signal.ITIMER_REAL, 0)
        if self.expired:
            self._bad(test)
            self.send("note", f"per-test timeout: {test.id()}")
        self.active = None
        self.send("end")
        super().stopTest(test)

    def _scope(self, holder):
        phase, _, suffix = holder.id().partition(" (")
        scope = self.by_scope[suffix.removesuffix(")")]
        if not scope:
            raise RuntimeError(f"unknown unittest fixture {holder.id()}")
        if phase.startswith("tearDown"):
            return [self.last] if self.last in scope else scope[-1:]
        return scope

    def _bad(self, test, err=None):
        if self.active is None:
            self.send("group", self._scope(test), True)
        else:
            self.send("fail", self.active)
        if self.failfast:
            self.stop()

    addFailure = addError = addUnexpectedSuccess = _bad

    def addSubTest(self, test, subtest, err):
        if err is not None:
            self._bad(test, err)
            if self.expired:
                raise TimedOut()

    def addSkip(self, test, reason):
        if self.active is None:
            self.send("group", self._scope(test), False)

    def addExpectedFailure(self, test, err):
        pass


class Suite(unittest.TestSuite):
    """Run unittest's native fixture handlers once per selected module/class."""

    @staticmethod
    def fixture(handler, result, scope, *args):
        result.send("scope", scope)
        signal.setitimer(signal.ITIMER_REAL, max(15.0, *(result.deadlines[i] for i in scope)))
        try:
            handler(*args, result)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)

    def run(self, result):
        for module, items in groupby(self, lambda t: type(t).__module__):
            if result.shouldStop:
                break
            cases = list(items)
            scope = result.by_scope[module]
            self.fixture(self._handleModuleFixture, result, scope, cases[0])
            if result._moduleSetUpFailed:
                continue
            try:
                for cls, items in groupby(cases, type):
                    if result.shouldStop:
                        break
                    members = list(items)
                    scope = result.by_scope[f"{cls.__module__}.{cls.__qualname__}"]
                    self.fixture(self._handleClassSetUp, result, scope, members[0])
                    result._previousTestClass = cls
                    if getattr(cls, "_classSetupFailed", False):
                        continue
                    try:
                        for test in members:
                            if result.shouldStop:
                                break
                            test(result)
                    finally:
                        self.fixture(self._tearDownPreviousClass, result, [result.last], None)
            finally:
                scope = result.by_scope[module]
                last = result.last if result.last in scope else scope[-1]
                self.fixture(self._handleModuleTearDown, result, [last])
                result._previousTestClass = None
        return result


def execute(ids, deadlines, early, send, case_filter=None):
    modules = {}
    for index, node in enumerate(ids):
        file, cls, method = node.split("::")
        module = file.removesuffix(".py").replace("/", ".")
        modules.setdefault(module, {}).setdefault(cls, []).append((index, method))

    result = Result(send, deadlines, early)

    def load(module, classes):
        indices = [idx for methods in classes.values() for idx, _ in methods]
        send("scope", indices)
        signal.setitimer(signal.ITIMER_REAL, max(15.0, *(deadlines[i] for i in indices)))
        try:
            imported = importlib.import_module(module)
        except unittest.SkipTest:
            send("group", indices, False)
            return []
        except (Exception, SystemExit) as exc:
            send("import", f"{module}: {type(exc).__name__}: {exc}")
            send("group", indices, True)
            if early:
                result.stop()
            return []
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        cases = []
        for cls, methods in classes.items():
            for idx, method in methods:
                try:
                    case = getattr(imported, cls)(method)
                except (AttributeError, TypeError, ValueError) as exc:
                    send("note", f"{ids[idx]}: {exc}")
                    send("group", [idx], True)
                    if early:
                        result.stop()
                        return cases
                    continue
                case._runner_index = idx
                if case_filter and ids[idx] in case_filter:
                    if ids[idx] == GENERATOR_MEMORY:
                        filter_dialects(case, imported, case_filter[ids[idx]])
                    elif ids[idx] == MAX_NODES:
                        filter_parser(case, case_filter[ids[idx]])
                    else:
                        filter_queries(case, imported, ids[idx], case_filter[ids[idx]])
                klass = type(case)
                result.by_scope[klass.__module__].append(idx)
                result.by_scope[f"{klass.__module__}.{klass.__qualname__}"].append(idx)
                cases.append(case)
        return cases

    # pytest imports all selected files during collection. With early exit,
    # collect a file only when its first method is reached.
    if early:
        for module, classes in modules.items():
            Suite(load(module, classes)).run(result)
            if result.shouldStop:
                break
    else:
        suite = Suite()
        for module, classes in modules.items():
            suite.addTests(load(module, classes))
        send("scope", [])
        suite.run(result)
