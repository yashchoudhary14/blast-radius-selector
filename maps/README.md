# SQLGlot baseline maps

These files describe the numbered tests at SQLGlot commit `eb11aec4be8cb2122afe713a4816662c3f0fb2fa`. Paths use forward slashes relative to the checkout. Test numbers come from the manifest's exact pytest node IDs. The skipped test has no number. Source line numbers start at 1.

| File | JSON format |
| --- | --- |
| `lines.json.gz` | `{"sets":[[test_number,...],...],"files":{"path.py":{"line":set_index,...},...}}`. The zero-based index selects a sorted, unique list of test numbers. Every package and test Python file has a key. A line without a numbered test has no key. |
| `modlevel.json.gz` | `{"path.py":[line,...],...}`. Sorted lines reached while importing the pinned files in module and class definition scopes. Includes executed decorators, default expressions and definition lines. An empty list is valid. |
| `reads.json.gz` | `{"path":[test_number,...],...}`. Each sorted list identifies tests that opened an existing file under the checkout with read access, including Python sources opened as data. |
| `durations.json` | `{"test_number":[wall_seconds,process_cpu_seconds],...}`. Each pair measures the complete pytest protocol in a separate serial run. Wall time includes waiting for child processes. Process CPU measures the pytest process, including its threads. |

## Rebuild

With the pinned `br-lab` Docker image available, run this from the Blast Radius track directory in PowerShell.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File lab/coverage/rebuild.ps1
```

The script runs `entry/build_coverage.py` with four CPUs in the pinned `br-lab` image. Its only input from the dataset is the test manifest. It replaces `lines.json.gz` after checking that every numbered test has coverage. A separate container runs the sample check. Build the selector image with `docker build -t br-j entry` to compile the function index and line caches.

The collector can also be invoked directly from the track directory.

```powershell
docker run --rm --pull=never --network=none --cpuset-cpus=0-3 -e SKIP_INTEGRATION=0 -e PYTHONPATH=/collector:/sqlglot -v "${PWD}/entry/build_coverage.py:/builder/build_coverage.py:ro" -v "${PWD}/lab/coverage:/collector:ro" -v "${PWD}/entry/maps:/maps" -v "${PWD}/sample/sample/tests.json:/manifest/tests.json:ro" br-lab python /builder/build_coverage.py
```

The collector runs the complete suite twice with coverage.py's C tracer and `SKIP_INTEGRATION=0`. The normal run uses pytest's usual import order. Before pytest collection in the second run, the builder imports every SQLGlot source module and instantiates each dialect's tokenizer, parser and generator under the unnumbered `<import>` context. The CLI receives an empty SQL argument and temporary version metadata for its argument parser.

Each numbered context spans setup through teardown. Shared fixture contexts expand to their class or module. Python children inherit the active context, and process-pool tasks carry it into reused workers. A subtest failure or setup/teardown error takes precedence over a passing call report. If a numbered test passes normally but fails or skips after preload, the builder prints its node ID and takes all its line coverage, including shared fixtures and child processes, from the normal run.

The separate file-read and import maps retain their original collection order. The original cold collector in `lab/coverage/build_maps.py` also measures durations and module-definition lines.

`lab/coverage/check_recall.py` reports the union size and missing test numbers for each sample commit. It uses exact old lines for deletions and replacements. For added lines it includes the closest covered line in the same revised function with a corresponding old coordinate, including both neighbors on a tie. It unions file readers for non-Python paths. Precision and recall use total commit–test pairs. Module-level impact beyond measured execution and changes to previously unexecuted paths can produce misses.
