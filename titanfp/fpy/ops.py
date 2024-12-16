"""Operators in the FPy language."""

from typing import Type

from ..fpbench import fpcast as fpc
from .fpyast import *

class FPyOp:
    """FPy operator metadata"""
    name: str
    arity: Optional[int]
    fpy: type
    fpc: type

    def __init__(self, name: str, arity: Optional[int], fpy: type, fpc: type):
        self.name = name
        self.arity = arity
        self.fpy = fpy
        self.fpc = fpc

_op_table: dict[str, FPyOp] = {}

def op_is_defined(name: str):
    """Returns whether or not `name` is an FPy operator."""
    return name in _op_table

def op_info(name: str):
    """Returns the metadata of an FPy operator or `None` if it does not exist."""
    return _op_table.get(name, None)

def _register_op(fpy: Type[NaryExpr], fpc: Type[fpc.NaryExpr], arity: Optional[int]):
    _op_table[fpy.name] = FPyOp(fpy.name, arity, fpy, fpc)

# unary operators
_register_op(Neg, fpc.Neg, 1)
_register_op(Fabs, fpc.Fabs, 1)
_register_op(Sqrt, fpc.Sqrt, 1)
_register_op(Cbrt, fpc.Cbrt, 1)
_register_op(Ceil, fpc.Ceil, 1)
_register_op(Floor, fpc.Floor, 1)
_register_op(Nearbyint, fpc.Nearbyint, 1)
_register_op(Round, fpc.Round, 1)
_register_op(Trunc, fpc.Trunc, 1)
_register_op(Acos, fpc.Acos, 1)
_register_op(Asin, fpc.Asin, 1)
_register_op(Atan, fpc.Atan, 1)
_register_op(Cos, fpc.Cos, 1)
_register_op(Sin, fpc.Sin, 1)
_register_op(Tan, fpc.Tan, 1)
_register_op(Acosh, fpc.Acosh, 1)
_register_op(Asinh, fpc.Asinh, 1)
_register_op(Atanh, fpc.Atanh, 1)
_register_op(Cosh, fpc.Cosh, 1)
_register_op(Sinh, fpc.Cosh, 1)
_register_op(Tanh, fpc.Tanh, 1)
_register_op(Exp, fpc.Exp, 1)
_register_op(Exp2, fpc.Exp2, 1)
_register_op(Expm1, fpc.Expm1, 1)
_register_op(Log, fpc.Log, 1)
_register_op(Log10, fpc.Log10, 1)
_register_op(Log1p, fpc.Log1p, 1)
_register_op(Log2, fpc.Log2, 1)
_register_op(Erf, fpc.Erf, 1)
_register_op(Erfc, fpc.Erfc, 1)
_register_op(Lgamma, fpc.Lgamma, 1)
_register_op(Tgamma, fpc.Tgamma, 1)
_register_op(IsFinite, fpc.Isfinite, 1)
_register_op(IsInf, fpc.Isinf, 1)
_register_op(IsNan, fpc.Isnan, 1)
_register_op(IsNormal, fpc.Isnormal, 1)
_register_op(Signbit, fpc.Signbit, 1)
_register_op(Not, fpc.Not, 1)
# binary
_register_op(Add, fpc.Add, 2)
_register_op(Sub, fpc.Sub, 2)
_register_op(Mul, fpc.Mul, 2)
_register_op(Div, fpc.Div, 2)
_register_op(Copysign, fpc.Copysign, 2)
_register_op(Fdim, fpc.Fdim, 2)
_register_op(Fmax, fpc.Fmax, 2)
_register_op(Fmin, fpc.Fmin, 2)
_register_op(Fmod, fpc.Fmod, 2)
_register_op(Remainder, fpc.Remainder, 2)
_register_op(Hypot, fpc.Hypot, 2)
_register_op(Atan2, fpc.Atan2, 2)
_register_op(Pow, fpc.Pow, 2)
# ternary
_register_op(Fma, fpc.Fma, 3)
# n-ary
_register_op(Or, fpc.Or, None)
_register_op(And, fpc.And, None)
