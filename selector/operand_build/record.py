"""Pure operand observations. The low ten bits are truth, the next ten are validity."""
import builtins
import enum
import math
import operator
import sys
from collections import defaultdict
from functools import lru_cache

RELATIONS = (operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge,
             operator.is_, operator.is_not, lambda a, b: a in b, lambda a, b: a not in b)
SCALARS = (type(None), bool, int, float, str, bytes)
MISSING = object()
vectors = defaultdict(set)
bounds = {}


def plain(cls):
    return type(cls) is type or type(cls) is enum.EnumType


@lru_cache(None)
def atom(cls):
    if cls in SCALARS:
        return True
    if type(cls) is enum.EnumType:
        return all(getattr(cls, name) is getattr(str, name)
                   for name in ("__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",
                                "__hash__", "__contains__"))
    return False


def scalar(value):
    cls = type(value)
    return plain(cls) and atom(cls)


def sequence(value):
    return (type(value) is tuple or type(value) is frozenset) and len(value) <= 32 and all(
        scalar(item) for item in value)


def pure_eq_none(value):
    cls = type(value)
    if not plain(cls):
        return False
    if cls.__eq__ is object.__eq__ and cls.__ne__ is object.__ne__:
        return True
    # Expr.__eq__ exits on type mismatch, before its hash-cache write.
    core = sys.modules.get("sqlglot.expressions.core")
    expr = vars(core).get("Expr") if core else None
    return expr is not None and cls.__eq__ is expr.__eq__ and cls.__ne__ is expr.__ne__


def relation(a, b):
    # Truth and validity are recorded independently. An exception or an unknown
    # implementation cannot prove equality, including when both sides raise.
    mask = 192
    truth = 64 if a is b else 128
    sa, sb = scalar(a), scalar(b)
    if (sa and sb) or (sa or sequence(a)) and (sb or sequence(b)):
        mask |= 3
        truth |= 1 if a == b else 2
        try:
            truth |= (a < b) * 4 | (a <= b) * 8 | (a > b) * 16 | (a >= b) * 32
            mask |= 60
        except TypeError:
            pass
    elif (a is None and pure_eq_none(b)) or (b is None and pure_eq_none(a)):
        mask |= 3
        truth |= 2
    if sa and (type(b) is str or type(b) is bytes or sequence(b)):
        try:
            truth |= 256 if a in b else 512
            mask |= 768
        except (TypeError, ValueError):
            pass
    return truth | (mask << 10)


def observe_bound(key, value):
    if (type(value) not in (str, int, float) or type(value) is str and len(value) > 96
            or type(value) is float and not math.isfinite(value)):
        bounds[key] = None
        return
    if key not in bounds:
        bounds[key] = (value, value)
    else:
        old = bounds[key]
        if old is not None:
            bounds[key] = ((min(old[0], value), max(old[1], value))
                           if type(old[0]) is type(value) else None)


def compare(key, op, side, a, b):
    rows = vectors[key]
    try:
        result = RELATIONS[op](a, b)
    except BaseException:
        rows.add(0)
        raise
    # A single unknown outcome already precludes all proofs at this site.
    if 0 not in rows:
        val = relation(a, b)
        if type(result) is not bool or not val & (1024 << op) or bool(val & (1 << op)) != result:
            rows.add(0)
        else:
            rows.add(val)
            if side:
                observe_bound(key, a if side == 1 else b)
    return result


def mark(key):
    vectors[key].add(0)


def pure_truth(value):
    cls = type(value)
    return (scalar(value) or (plain(cls) and (cls in (tuple, list, dict, set, frozenset) or
            not hasattr(cls, "__bool__") and not hasattr(cls, "__len__"))))


def boolean(key, left, is_name, arg, condition):
    rows = vectors[key]
    if 0 not in rows:
        right = sys._getframe(1).f_locals.get(arg, MISSING) if is_name else arg
        # A local read or literal can be elided without side effects. In a
        # value context the replacement must return the very same object.
        safe = (right is not MISSING and pure_truth(left) and pure_truth(right) and
                (left is right or condition and bool(left) == bool(right)))
        rows.add(1 if safe else 0)
    return left


def install():
    builtins._br_operand_compare = compare
    builtins._br_operand_boolean = boolean
    builtins._br_operand_mark = mark
