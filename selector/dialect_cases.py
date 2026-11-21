"""Select generator-memory cases from pristine, independently observed imports."""

import ast
import json
from pathlib import Path

from .source_equivalence import same_directives


def isolated_test_methods(original, updated):
    """Check that an unrelated test module has no changed import-time code."""
    def shell(source):
        tree = ast.parse(source)
        for cls in tree.body:
            if not (isinstance(cls, ast.ClassDef) and not cls.decorator_list and not cls.keywords
                    and len(cls.bases) == 1 and isinstance(cls.bases[0], ast.Name)
                    and cls.bases[0].id == "Validator"):
                continue
            cls.body = [
                stmt for stmt in cls.body
                if not (isinstance(stmt, ast.FunctionDef) and stmt.name.startswith("test_")
                        and not stmt.decorator_list and not stmt.returns
                        and not getattr(stmt, "type_params", ())
                        and len(stmt.args.args) == 1 and stmt.args.args[0].arg == "self"
                        and not stmt.args.args[0].annotation and not stmt.args.posonlyargs
                        and not stmt.args.kwonlyargs and not stmt.args.defaults
                        and not stmt.args.vararg and not stmt.args.kwarg)
            ]
        return ast.dump(tree)

    try:
        return shell(original) == shell(updated) and same_directives(original, updated)
    except (SyntaxError, ValueError):
        return False


class DialectCases:
    def __init__(self, maps):
        data = json.loads((Path(maps) / "dialect_closure.json").read_bytes())
        self.common = set(data["common"])
        self.cases = {row["name"]: set(row["paths"]) for row in data["cases"]}
        self.cpu = {row["name"]: row["cpu_s"] for row in data["cases"]}

    def subset(self, candidates):
        paths = [path for path in candidates.python_paths if path.startswith("sqlglot/")]
        if (not paths or candidates.broken or not candidates.dialect_subset_safe
                or self.common.intersection(paths)):
            return None
        names = [name for name, files in self.cases.items() if not files.isdisjoint(paths)]
        return names if len(names) < len(self.cases) else None

    def body_cost(self, names):
        return sum(self.cpu[name] for name in names)
