"""Identify edits confined to ordinary unittest methods in pinned test modules."""

import ast


def _method(node):
    if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
        return False
    args = node.args
    return (not node.decorator_list and not getattr(node, "type_params", ())
            and not node.returns and not node.type_comment and not args.posonlyargs
            and not args.kwonlyargs and not args.defaults and not args.vararg
            and not args.kwarg and len(args.args) == 1 and args.args[0].arg == "self"
            and not args.args[0].annotation)


def _test_class(node, validator):
    if not isinstance(node, ast.ClassDef) or not node.name.startswith("Test"):
        return False
    if node.decorator_list or node.keywords or len(node.bases) != 1:
        return False
    base = node.bases[0]
    return ((validator and isinstance(base, ast.Name) and base.id == "Validator")
            or (isinstance(base, ast.Attribute) and base.attr == "TestCase"
                and isinstance(base.value, ast.Name) and base.value.id == "unittest"))


def method_changes(old, new, *, local_validator=False):
    """Return changed qualified names if all AST differences are plain tests."""
    def methods(tree):
        validator = local_validator or any(isinstance(node, ast.ImportFrom) and node.level == 0
                        and node.module == "tests.dialects.test_dialect"
                        and any(alias.name == "Validator" and alias.asname is None for alias in node.names)
                        for node in tree.body)
        found = {}
        for cls in tree.body:
            if not _test_class(cls, validator):
                continue
            names = [node.name for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
            for node in cls.body:
                if _method(node) and names.count(node.name) == 1:
                    found[cls.name + "." + node.name] = node
        return found

    def project(tree, own, other):
        removed = {own[name] for name in own.keys() - other.keys()}
        for name in changes & own.keys() & other.keys():
            own[name].body = [ast.Pass()]
        for cls in tree.body:
            if isinstance(cls, ast.ClassDef):
                doc = ast.get_docstring(cls, clean=False)
                cls.body = [node for node in cls.body if node not in removed]
                if ast.get_docstring(cls, clean=False) != doc:
                    return None
        return ast.dump(tree)

    before, after = ast.parse(old), ast.parse(new)
    left, right = methods(before), methods(after)
    changes = left.keys() ^ right.keys()
    changes.update(name for name in left.keys() & right.keys() if ast.dump(left[name]) != ast.dump(right[name]))
    projected = project(before, left, right)
    return changes if projected is not None and projected == project(after, right, left) else None
