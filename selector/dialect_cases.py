"""Select generator-memory cases from pristine, independently observed imports."""

import json
from pathlib import Path


class DialectCases:
    def __init__(self, maps):
        data = json.loads((Path(maps) / "dialect_closure.json").read_bytes())
        self.common = set(data["common"])
        self.cases = {row["name"]: set(row["paths"]) for row in data["cases"]}

    def subset(self, candidates):
        paths = candidates.python_paths
        if (not paths or candidates.broken or not candidates.dialect_subset_safe
                or self.common.intersection(paths)):
            return None
        names = [name for name, files in self.cases.items() if not files.isdisjoint(paths)]
        return names if len(names) < len(self.cases) else None
