"""Sites where the pinned SQLGlot dispatches on parser or expression keys."""

SITES = {
    "sqlglot/parser.py": {
        "self.FUNCTION_PARSERS.get(upper)": ("FUNCTION_PARSERS",),
        "functions.get(upper)": ("FUNCTIONS",),
        "self._curr.text.upper() in texts": ("PROPERTY_PARSERS",),
        "self._curr.token_type in types": ("RANGE_PARSERS",),
        "self._curr.token_type in self.RANGE_PARSERS": ("RANGE_PARSERS",),
    },
    "sqlglot/expressions/builders.py": {
        "dialect.parser_class.FUNCTIONS.get(name.upper())": ("FUNCTIONS",),
    },
    "sqlglot/parsers/prql.py": {
        "self.FUNCTIONS.get(name)": ("FUNCTIONS",),
    },
    "sqlglot/generator.py": {
        "self._dispatch.get(expression.__class__)": ("TRANSFORMS",),
        "self.TRANSFORMS.get(expression.__class__)": ("TRANSFORMS",),
        "getattr(self, f'{node.key}_sql')": ("GENERATOR_METHOD",),
    },
    "sqlglot/transforms.py": {
        "self.TRANSFORMS.get(type(expression))": ("TRANSFORMS",),
        "getattr(self, expression.key + '_sql', None)": ("GENERATOR_METHOD",),
    },
    "sqlglot/optimizer/annotate_types.py": {
        "self.expression_metadata.get(expr.__class__)": ("EXPRESSION_METADATA",),
    },
    "sqlglot/serde.py": {
        "getattr(module, class_name)": ("MODULE_ATTRIBUTE",),
    },
}
