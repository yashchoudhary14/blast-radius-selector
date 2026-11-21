# Patched sqlglot test runner

`Session` runs selected unittest methods on the pinned Linux checkout. It uses Python's standard library. Keep one session for a sequence of independent patches.

```python
from runner import Session

with Session(base="/sqlglot", tmp="/tmp/br") as session:
    root = session.apply_only(patch_text)
    source = (root / "sqlglot/parser.py").read_bytes()
    result = session.run(patch_text, test_ids, timeout_factor=10,
                         per_test_budget=recorded_seconds,
                         stop_on_first_fail=False, overall_timeout=30)
```

Read patch files with `path.read_bytes().decode("utf-8", "surrogateescape")` to preserve literal carriage returns in hunk lines. The default session constructor loads catalog timings and the unittest machinery. Project imports start in the first child, or during the first fixture-only preload.

- `test_ids` contains strings such as `tests/test_parser.py::TestParser::test_ref_type`. The runner groups selections by module and class in first-seen order. Each selected method runs once.
- `case_filter={GENERATOR_MEMORY: ["BIGQUERY", "DREMIO"]}` runs the named enum members in their original order. Import `GENERATOR_MEMORY` from `runner`. The child restores the original enum before teardown. A failed subtest marks the method as failed.
- `per_test_budget` maps IDs to **recorded seconds**. Overrides take precedence over `/data/tests.json`, which the session reads once if present. Each test gets `max(1.0, timeout_factor * recorded_seconds)` wall seconds, including its per-method setup and teardown. IDs without a recorded duration get one second.
- `failing` is a set of IDs. `ran` lists attempted methods and IDs resolved by a shared import or fixture outcome. `import_error` contains the first import diagnostic. `notes` includes additional import errors, timeouts and interrupted runs.
- `stop_on_first_fail=True` collects one file at a time and stops at the first failure. A shared setup or import error labels all selected IDs in its scope. An unexpected success or a failure in any subtest labels its parent method.
- `overall_timeout` limits fork and execution wall time. Its default is 15 seconds plus the sum of method deadlines. If the process group hits that guard, the parent preserves completed results, marks the active scope as failing and reports the interruption. IDs absent from `ran` are unresolved.
- `cpu_s` counts child user and system CPU using the parent's `RUSAGE_CHILDREN` delta. The runner reaps helpers in the test process group, including descendants orphaned on early return or timeout. A selector worker below PID 1 temporarily enables Linux child-subreaper mode. Parent staging and bookkeeping CPU are additional. Use sessions sequentially when measuring CPU this way.
- `apply_only` returns a `pathlib.Path` valid until the next operation or `close()`. `run` removes its overlay before returning. The applier requires a textual unified diff. A mismatched hunk, Git binary patch or metadata-only rename raises `runner.PatchError`.

The overlay symlinks untouched entries to the base checkout. A directory containing changed Python files exposes only the unaffected bytecode caches. The runner also gives the pinned test that resolves its own source path a local inode when the sibling `sqlglotc` package changes. Python subprocesses inherit the overlay import path. Each test child uses at most two available CPUs so process pools respect the container quota.

For data-only changes under `/sqlglot/tests/fixtures` (`.sql`, `.json`, `.gz`), the session lazily imports pristine sqlglot, the test helpers and the parser test module through a stable overlay path. Only the test children receive the saved module objects. The parent restores the caller's module table. Python changes always use fresh project imports. `Session(warm_fixtures=False)` selects the cold reference path.

## Reproduction

From the track directory on Windows, with the pinned `br-lab` image available, run

```powershell
& 'C:\dev\Buildathon\.venv\Scripts\python.exe' entry/runner/check.py --jobs 5
```

This builds `br-runner`, runs the regression tests and import benchmark, and checks the sample and any labeled `lab/corpus/c1`. For one additional dataset, supply `--data lab/corpus/c1`. Each shard has two CPUs, a 4 GiB memory limit, 256 PIDs, a read-only root, a 64 MiB tmpfs, user 65534, `HOME=/tmp` and disabled networking. The full offline comparison spans several container runs. Per-commit timings and mismatched IDs are written to `scratch/runner/<dataset>/*.jsonl`.

While corpus labels are being generated, `entry/runner/corpus.py --jobs 5` records the full failing sets and child timings in `scratch/runner/corpus/c1`. Rerun with `--compare-only` after `answers.json` appears. The cache checks the patch digest, runner source and numbered catalog before reusing an observation. An incomplete child run fails verification even if every requested ID appears in its partial result.

## Import measurements

The benchmark collects one parser test and omits its body. The first-call column includes parent staging and any one-time preload. The other columns give the median of the next seven calls in seconds.

| Patch and import strategy | First call, total CPU | Fork + collection, child CPU | Total CPU |
| --- | ---: | ---: | ---: |
| Comment added to `sqlglot/parser.py`, fresh imports | 0.2223 | 0.1775 | 0.1886 |
| SQL fixture, fresh imports | 0.0749 | 0.0699 | 0.0731 |
| SQL fixture, retained parser and helper imports | 0.0698 | 0.0048 | 0.0088 |
| SQL fixture, experimental preload of every numbered test module | 0.6022 | 0.0096 | 0.0154 |

The 50-test benchmark samples the catalog with `random.Random(42)` and restores catalog order. Its recorded duration is 0.720 s. The full catalog has 1,620 tests and 103.047 recorded seconds.

## Sample exactness and cost

All 30 full failing sets matched the organizers' answers. The runner attempted all 48,600 methods and identified all 73 labeled failures with zero false positives. All 30 subset runs also matched, totaling 1,500 decisions. The nine regression tests cover patch application, stale bytecode, spawned interpreters, fixture and subtest failures, unexpected success, skips, deadlines, early exit and switching between cached fixture and fresh code imports.

| Selection per commit | Mean child CPU | Median child CPU | Child CPU / recorded seconds | Mean parent CPU |
| --- | ---: | ---: | ---: | ---: |
| 1,620 tests | 143.493 s | 142.952 s | 1.393× | 0.156 s |
| 50 tests | 3.184 s | 3.034 s | 4.422× | 0.024 s |

The local Docker Desktop host also ran other lab jobs during the measurement. Each cell below is child CPU seconds for a single commit.

| Sample commit | 1,620 tests | 50 tests |
| --- | ---: | ---: |
| 2haz | 110.717 | 2.490 |
| 2jrq | 98.882 | 2.477 |
| 2qn5 | 110.052 | 2.538 |
| 2vdj | 111.345 | 2.459 |
| 3utz | 71.737 | 2.507 |
| 4d63 | 175.183 | 2.528 |
| 535z | 172.569 | 2.809 |
| 5y72 | 140.262 | 2.585 |
| 66ks | 143.974 | 2.668 |
| 6vvp | 156.818 | 2.551 |
| 6ze8 | 162.939 | 3.263 |
| 8y7j | 136.970 | 3.057 |
| 986c | 160.377 | 3.886 |
| 9u5k | 135.509 | 3.759 |
| ary6 | 131.512 | 3.607 |
| bap8 | 159.631 | 3.326 |
| bj7g | 136.806 | 3.010 |
| bvw2 | 157.346 | 2.921 |
| egys | 136.542 | 3.161 |
| f4bn | 140.310 | 3.107 |
| f4xa | 161.793 | 4.618 |
| fryt | 166.308 | 2.935 |
| gbav | 171.710 | 2.897 |
| gbhk | 133.866 | 3.406 |
| j6uz | 160.757 | 4.575 |
| ktz3 | 132.572 | 3.283 |
| mthh | 160.221 | 3.113 |
| nwnx | 141.929 | 2.868 |
| rszg | 165.641 | 3.783 |
| st8f | 160.516 | 5.331 |
