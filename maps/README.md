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

The script mounts the source checkout and collector read-only, pins the container to four CPUs and writes the maps to `entry/maps/`. The builder receives only the test manifest. A separate verification container reads the sample patches and answers after the build.

The collector runs pytest twice with integration tests enabled. The first run uses coverage.py 7.13.1 with its C tracer, a node-ID context spanning setup through teardown, and a Python `open` audit hook. Class- and module-scoped fixture setup has its own context, which expands to every numbered test in that scope. Python children inherit the active context. Each submitted process-pool task also carries its own context so argument decoding, test work and result encoding in reused workers can be attributed. The builder combines all process coverage databases before interning test sets. The second run measures durations without tracing or an audit hook.

The builder imports every source in a fresh interpreter state for SQLGlot, measures executed lines with coverage.py, and intersects them with the compiled AST's line tables for definition scopes. Collection-time imports have no numbered test context. Lazy imports can appear under the first test that triggers them. In the pinned development checkout, `sqlglot/__main__.py` stops at the absent generated version and `tests/gen_fixtures.py` stops at an unavailable external fixture directory. Their entries include lines reached before those exceptions.

`lab/coverage/check_recall.py` reports the union size and missing test numbers for each sample commit. It uses exact old lines for deletions and replacements. For added lines it includes the closest covered line in the same revised function with a corresponding old coordinate, including both neighbors on a tie. It unions file readers for non-Python paths. Precision and recall use total commit–test pairs. Module-level impact beyond measured execution and changes to previously unexecuted paths can produce misses.
