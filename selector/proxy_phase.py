"""Phase-boundary values for the five pinned tests. Imported only by the probe child."""
import enum
import json
import pickle


def expression_state(root):
    from sqlglot.serde import dump
    return [dump(root), object_graph(root)]


def object_graph(root):
    from sqlglot import exp

    # serde is the wire representation used by Expression.__reduce__. Also preserve
    # the live object graph that Plan receives, including aliases and stale caches.
    nodes, objects, seen = [], [], {}

    def ref(value):
        cls = type(value)
        if value is None or cls in (bool, int, float, str):
            return value
        if isinstance(value, enum.Enum):
            return ["enum", cls.__module__, cls.__qualname__, value.name]
        index = seen.get(id(value))
        if index is None:
            index = seen[id(value)] = len(objects)
            objects.append(value)
        return ["ref", index]

    ref(root)
    for obj in objects:
        cls = type(obj)
        if isinstance(obj, exp.Expr):
            slots = [ref(getattr(obj, key)) for key in exp.Expression.__slots__]
            attrs = [[key, ref(v)] for key, v in vars(obj).items()]
            nodes.append([cls.__module__, cls.__qualname__, slots, attrs])
        elif cls is dict:
            nodes.append(["dict", [[ref(k), ref(v)] for k, v in obj.items()]])
        elif cls in (tuple, list):
            nodes.append([cls.__name__, [ref(v) for v in obj]])
        else:
            raise ValueError(f"unsupported expression state {cls}")
    return nodes


def transfer(value):
    # multiprocessing.reduction.ForkingPickler uses the default pickle protocol.
    from multiprocessing.reduction import ForkingPickler
    return pickle.loads(ForkingPickler.dumps(value))


def encode(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")


def replay(number, inputs, emit):
    from sqlglot import find_tables, parse_one, transpile
    from sqlglot.optimizer import optimize
    from sqlglot.schema import ensure_schema
    from tests.helpers import TPCDS_SCHEMA, TPCH_SCHEMA, load_sql_fixture_pairs

    if number in (1106, 1107, 1123):
        bench = "ds" if number == 1106 else "h"
        pairs = list(load_sql_fixture_pairs(f"optimizer/tpc-{bench}/tpc-{bench}.sql"))
        if number == 1123:
            for _, sql, optimized in pairs:
                for query in (sql, optimized):
                    emit("sql", transpile(query, write="duckdb")[0])
            return
        schema = TPCDS_SCHEMA if bench == "ds" else TPCH_SCHEMA
        for meta, sql, _ in pairs:
            expr, meta = transfer((parse_one(sql), meta))
            emit("worker", [meta, expression_state(expr)])
            if meta.get("execute"):
                # find_tables hashes table nodes before execute() calls optimize.
                names = [table.name for table in find_tables(expr)]
                emit("tables", sorted(set(names)))
                optimized = optimize(expr, ensure_schema(schema), leave_tables_isolated=True, dialect=None)
                emit("optimized", expression_state(optimized))
        for meta, sql, _ in pairs:
            if meta.get("execute"):
                emit("sql", transpile(sql, write="duckdb")[0])
    elif number == 1082:
        sql, schema = inputs
        emit("optimized", expression_state(optimize(sql, schema=schema)))
    elif number == 1361:
        # Preserve the full shared setUp schema, including nullable DataType objects.
        import sqlglot
        from sqlglot.schema import MappingSchema
        sqlglot.schema = MappingSchema()
        # This pinned method only needs the sqlglot exp module and self.
        from types import SimpleNamespace
        from sqlglot import exp
        setup = {}
        exec(compile(inputs["setup"], "<pinned TestOptimizer.setUp>", "exec"), {"exp": exp}, setup)
        holder = SimpleNamespace()
        setup["setUp"](holder)
        schema = holder.schema
        for sql in inputs["queries"]:
            optimized = optimize(sql, schema=schema, dialect="duckdb")
            emit("sql", sql)
            emit("sql", optimized.sql("duckdb"))
    else:
        raise ValueError(number)
