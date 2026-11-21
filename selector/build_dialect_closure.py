"""Build the generator-memory case map inside the pinned br-lab image."""

import json
import subprocess
import sys
import time
import unittest
from pathlib import Path


def closure(name):
    import tests.test_generator as module

    member = module.Dialects[name]
    root = Path("/sqlglot")
    module.Dialect.get_or_raise(member).generator()
    return sorted({
        path.relative_to(root).as_posix()
        for mod in sys.modules.copy().values()
        if (filename := getattr(mod, "__file__", None))
        and (path := Path(filename)).is_relative_to(root / "sqlglot")
        and path.suffix == ".py"
    })


def measure(name):
    import tests.test_generator as module

    module.Dialects = [module.Dialects[name]]
    case = module.TestGenerator("test_generator_memory")
    result = unittest.TestResult()
    wall, cpu = time.perf_counter(), time.process_time()
    case.run(result)
    cpu, wall = time.process_time() - cpu, time.perf_counter() - wall
    if not result.wasSuccessful():
        raise RuntimeError(result.errors + result.failures)
    return {"wall_s": round(wall, 6), "cpu_s": round(cpu, 6)}


def build(output):
    from tests.test_generator import Dialects

    def fresh(mode, name):
        return json.loads(subprocess.check_output([sys.executable, __file__, mode, name]))

    cases = [{"name": m.name, "paths": fresh("--closure", m.name), **fresh("--time", m.name)}
             for m in Dialects]
    common = set.intersection(*(set(case["paths"]) for case in cases))
    for case in cases:
        case["paths"] = sorted(set(case["paths"]) - common)
    output.write_text(json.dumps({"common": sorted(common), "cases": cases}, indent=2) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "--closure":
        print(json.dumps(closure(sys.argv[2])))
    elif sys.argv[1] == "--time":
        print(json.dumps(measure(sys.argv[2])))
    else:
        build(Path(sys.argv[1]))
