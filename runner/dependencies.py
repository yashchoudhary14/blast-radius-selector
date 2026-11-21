"""Build the reverse graph of modules loaded when each pristine module initializes."""
import importlib
import json
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = Path("/sqlglot")
MAPS = Path("/app/maps")


def inspect(name):
    importlib.import_module(name)
    return sorted({str(Path(module.__file__).relative_to(BASE))
                   for key, module in sys.modules.copy().items()
                   if key.partition(".")[0] == "sqlglot" and getattr(module, "__file__", None)})


def collect(path):
    parts = path.relative_to(BASE).with_suffix("").parts
    name = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
    output = subprocess.check_output([sys.executable, "-m", "runner.dependencies", name])
    return str(path.relative_to(BASE)), json.loads(output)


def build():
    imports = json.loads((MAPS / "collect_imports.json").read_bytes())
    paths = sorted(p for p in (BASE / "sqlglot").rglob("*.py") if p.name != "__main__.py")
    with ThreadPoolExecutor(4) as pool:
        imports.update(pool.map(collect, paths))
    # Include file reads during import as edges. Directory membership changes
    # outside the audited fixture tree are routed to the cold runner.
    for module, (reads, _) in json.loads((MAPS / "collection_reads.json").read_bytes()).items():
        imports[module].extend(reads)
    reverse = defaultdict(set)
    for importer, files in imports.items():
        for path in files:
            if path != importer:
                reverse[path].add(importer)
    (MAPS / "runner_importers.json").write_text(json.dumps({p: sorted(v) for p, v in reverse.items()}))


if __name__ == "__main__":
    if len(sys.argv) == 2:
        print(json.dumps(inspect(sys.argv[1])))
    else:
        build()
