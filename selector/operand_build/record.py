"""Pure operand observations. The low ten bits are truth, the next ten are validity."""
import builtins
import enum
import math
import operator
import sys
from types import FunctionType
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


def slot(cls, name):
    for base in cls.__mro__:
        if name in base.__dict__:
            return base.__dict__[name]
    return MISSING


@lru_cache(None)
def atom(cls):
    if cls in SCALARS:
        return True
    if type(cls) is enum.EnumType:
        return all(slot(cls, name) is getattr(str, name)
                   for name in ("__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",
                                "__hash__", "__contains__"))
    return False


def scalar(value):
    cls = type(value)
    return plain(cls) and atom(cls)


def sequence(value):
    return (type(value) is tuple or type(value) is frozenset) and len(value) <= 32 and all(
        scalar(item) for item in value)


def eq_none_mask(value):
    cls = type(value)
    if not plain(cls):
        return 0
    eq, ne = slot(cls, "__eq__"), slot(cls, "__ne__")
    if eq is object.__eq__:
        return 3 if ne is object.__ne__ else 1
    # Expression.__eq__ exits on type mismatch, before its hash-cache write.
    core = sys.modules.get("sqlglot.expressions.core")
    expr = vars(core).get("Expression") if core else None
    if (expr is None or eq is not vars(expr)["__eq__"] or type(eq) is not FunctionType
            or eq.__code__.co_filename != "/sqlglot/sqlglot/expressions/core.py"
            or eq.__code__.co_firstlineno != 837):
        return 0
    # Expression.__ne__ dynamically reads self.__eq__. Allow it only when
    # ordinary attribute lookup must resolve to the audited implementation.
    if (ne is vars(expr)["__ne__"] and type(ne) is FunctionType
            and ne.__code__.co_filename == eq.__code__.co_filename and ne.__code__.co_firstlineno == 840
            and cls.__dictoffset__ == 0
            and slot(cls, "__getattribute__") is object.__getattribute__):
        return 3
    return 1


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
    elif a is None or b is None:
        certified = eq_none_mask(b if a is None else a)
        mask |= certified
        truth |= certified & 2
    elif type(a) in (type, enum.EnumType) and type(b) in (type, enum.EnumType):
        mask |= 3
        truth |= 1 if a is b else 2
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
            slot(cls, "__bool__") is MISSING and slot(cls, "__len__") is MISSING)))


def boolean(key, left, is_name, arg, condition):
    rows = vectors[key]
    if 0 not in rows:
        right = arg
        if is_name:
            frame = sys._getframe(1)
            code = frame.f_code
            names = code.co_varnames + code.co_cellvars + code.co_freevars
            right = frame.f_locals.get(arg, MISSING) if arg in names else MISSING
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
