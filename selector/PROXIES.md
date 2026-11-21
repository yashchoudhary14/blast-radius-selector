# Phase-boundary proofs

`can_prove_pass(n, changed_spans, data_changed_modules, overlay_root)` returns `True` on a complete matching replay. `None` retains the candidate. An optional monotonic `deadline=` caps the subprocess wait. The maps describe SQLGlot `eb11aec4be8cb2122afe713a4816662c3f0fb2fa` under the locked CPython 3.13 environment.

## What the collector observes

The builder runs each actual unittest, including imports, class setup, teardown and assertions, in its own interpreter. It traces everything outside the wrapped phase-A calls into the protected line set. Observation code gets its own excluded coverage context. The executor pool runs inline with the real initializer, task and result, using multiprocessing's pickler in both directions. The build fails on any unsuccessful, missing or skipped baseline test. Collection requires coverage.py only in the lab.

| Test | Per-query values compared with the passing run |
| --- | --- |
| 1082, aggregation order | One optimized expression immediately before `Plan`. The original assertion inspects aliases in planner order. |
| 1106, execute TPC-DS | 99 deserialized worker inputs and metadata, 57 referenced-table name sets, 57 optimized expressions and 57 DuckDB reference SQL strings. The remaining 42 tasks exit at the metadata check. |
| 1107, execute TPC-H | All 22 expressions must parse, pickle and unpickle. None of the 22 pinned fixture entries has the execute flag. |
| 1123, optimized TPC-H | Both the original and optimized fixture go through DuckDB transpilation, giving 44 strings for 22 comparisons. |
| 1361, IN with NULLs | 180 pairs of original and optimized DuckDB SQL, including duplicates and ordering. Replay runs the pinned method setup to create its full schema. |

The expression witness contains the SQLGlot serde dump, including types, metadata and comments. Its second part encodes the live object graph with traversal-index references. It preserves class and enum identity by name, ordered argument keys, empty lists, explicit `None`, shared containers, parent links, argument keys, child indices, raw type pointers, instance attributes and cached hashes. Primitive values retain their JSON representations. A new unsupported object type aborts the proof. Each gzip stream holds one canonical value per line. Comparison includes the stream length and uses the complete bytes.

## Conditions for dropping a candidate

1. Supply every changed checkout-relative path and inclusive old-file statement intervals. An insertion with ambiguous ownership needs its enclosing old function or an abstention. The API checks the runner's symlink overlay against the base and requires its changed leaves to equal the reported file set. It declines test, fixture, non-package, added and deleted files. It also declines any reported class or module data change, even for modules outside the measured coverage.
2. Each reported interval must be disjoint from that test's protected lines. Those lines include executor initialization, table registration, the optional schema-inference branch when taken, the planner, Python SQL generation and execution, reference-result comparisons, class fixtures and test imports.
3. In a fresh interpreter the overlay is the only SQLGlot import root. Replay uses the real parser, serializer, optimizer and transpiler. For executor work it also runs `find_tables` before optimization and passes the same ensured schema, explicit dialect and `leave_tables_isolated=True`.
4. Equal SQL or ASTs alone cannot establish that a patch left shared state intact. The import guard compares the complete old and revised module ASTs. It accepts a limited family of existing expression changes inside function bodies. A reached identity/equality substitution must compute the same Boolean on the same pair of operands using vetted built-in comparison implementations. Each operand is evaluated once. Other supported substitutions abort on first execution, even if application code catches the exception. Unsupported syntax, changed signatures, initializers, protocol methods, new bindings, calls and state writes cause abstention. This guard is intentionally conservative on equivalent edits.

For this pinned deterministic workload, consider the expensive continuation at each boundary. Its checkout data is unchanged, it receives the same observed inputs, and the patch changed no statement on its baseline path. The expression guard also prevents recognized edits in replayed code from silently changing mutable state before the cut. Under these conditions replay of the continuation gives the same results and assertions. Unknown cases retain the test.

## State, scheduling and deadlines

- The baseline and probe fix `PYTHONHASHSEED=0` so cached expression hashes can be checked verbatim. The fixtures contain no time-dependent or random SQL operations. The pinned tests determine where result order matters. The witness preserves index and sequence order throughout.
- Each probe starts with fresh module globals, dialect registries and optimizer caches. No probe interpreter or patched module survives between commits. Fixture cache hits can reuse only previously computed reference SQL for the same query. Raw worker messages are round-tripped before optimization, and both sides of DuckDB comparisons are captured.
- The inline pool models the pinned schema, tables and serialized tasks. This argument assumes equivalent cold/warm worker behavior for these deterministic queries. Exhaustive equivalence across arbitrary native scheduling, hash seeds, suite-global interference and external infrastructure failures cannot be established from a finite baseline trace. A change introducing a process/time/randomness dependency, arbitrary state mutation or initialization edit is outside the admitted source transformations.
- Exceptions, unsupported values, incomplete streams, a missing map, signal termination and wall-deadline expiry all retain the test. The parent kills and waits for the probe on timeout. A successful probe does not predict machine-dependent timeouts on an otherwise unchanged full test.

## Selector hook

After the cheap-test decisions, stage the overlay once. `selected` is the current set, `eligible` contains the function-coverage candidates, `millis` holds the catalog durations and `deadline` is the existing lane deadline. `spans` and `data_changed_modules` must describe the entire patch. The initial pristine probe CPU measurements rounded up to tenths of a second give `cost = {1082: 0.2, 1106: 3.9, 1107: 0.3, 1123: 0.4, 1361: 1.3}`.

```python
from .proxies import TESTS, can_prove_pass

root = session.apply_only(patch)
for n in sorted(selected.intersection(TESTS, eligible), key=millis.get, reverse=True):
    gain_ms = schedule(selected, millis)[1] - schedule(selected - {n}, millis)[1]
    if gain_ms > 1000 * cost[n] and time.monotonic() + cost[n] < deadline:
        if can_prove_pass(n, spans, data_changed_modules, root, deadline=deadline):
            selected.remove(n)
```

The gain is the change in the entire worker schedule. Removing a 21.454-second test can save very little when other selected tests determine the busiest load. The threshold uses the maximum available saving. An attempt can still cost CPU and retain the test.

## Verification

The gate scanned 30 sample patches, 60 c1 patches and all 300 c2 patches. It used the baseline union of tests covering each structurally changed function, excluding import-only definition lines as in the selector index. All 180 candidate decisions had ground truth. The sample and c1 used answers.json. For c2, 224 completed full-suite reports were available at comparison time. The other 33 candidate labels came from independent runs of the actual methods on the matching patches. The gate prefers the full-suite answers when they become available.

| Test | Proved, passed | Proved, failed | Retained, passed | Retained, failed | Pristine proof CPU s | Recorded test s | All attempts CPU s | Avoided recorded s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1082 | 4 | 0 | 20 | 3 | 0.289 | 1.364 | 6.156 | 5.456 |
| 1106 | 8 | 0 | 42 | 15 | 6.323 | 21.454 | 65.256 | 171.632 |
| 1107 | 2 | 0 | 16 | 4 | 0.476 | 3.863 | 5.707 | 7.726 |
| 1123 | 3 | 0 | 16 | 6 | 0.444 | 3.537 | 8.729 | 10.611 |
| 1361 | 8 | 0 | 27 | 6 | 1.637 | 1.331 | 19.255 | 10.648 |
| Total | 25 | 0 | 121 | 34 | | | 105.103 | 206.073 |

CPU includes probe setup, the parent guard and the fresh child. The separate pristine benchmark varied with machine load. Two runs of the full TPC-DS proof measured 3.825 and 6.323 CPU seconds. The integration estimates above use the earlier run. The 206.073 seconds are the sum of catalog durations for 25 avoided inclusions.

| Dataset | Candidates | Proven passes | False passes | Actual failures retained |
| --- | ---: | ---: | ---: | ---: |
| Sample | 7 | 2 | 0 | 3 |
| c1 | 27 | 7 | 0 | 3 |
| c2 | 146 | 16 | 0 | 28 |

The local scorer ran the same static selector and changed-function analysis in each treatment under the official container flags. Each cell gives simulated seconds / CPU seconds / rounded total score. All six evaluations had zero missed commits.

| Dataset | Before | Probe every eligible test | Gate by marginal worker saving |
| --- | ---: | ---: | ---: |
| Sample | 275.334 / 1.418 / 276.8 | 262.295 / 8.246 / 270.5 | 262.566 / 7.891 / 270.5 |
| c1 | 750.150 / 9.467 / 759.6 | 748.150 / 32.061 / 780.2 | 750.150 / 7.169 / 757.3 |

The budgeted c1 treatment saved zero simulated seconds. Its total-score difference reflects CPU variation on the shared host. These measurements use one run per treatment. Production lane timeouts and cheap-test removal can change the marginal worker saving.

From the track directory, this rebuilds the maps, checks seven fail-closed cases, gates all three datasets and runs the local evaluator on sample and c1. The lab wrapper requires the local pinned br-lab, br-entry and br-entry-static images. It stores individual evaluation decisions under scratch/proxy.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File lab/coverage/proxy_check.ps1 all
```
