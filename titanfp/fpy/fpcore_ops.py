"""Tables for the FPy->FPCore compiler."""

from typing import Callable, Type

from ..fpbench import fpcast as fpc
from .fpyast import *

known_table : dict[Type[NaryExpr], Callable[..., fpc.Expr]] = {
    # unary
    Neg : fpc.Neg,
    Fabs : fpc.Fabs,
    Sqrt : fpc.Sqrt,
    Cbrt : fpc.Cbrt,
    Ceil : fpc.Ceil,
    Floor : fpc.Floor,
    Nearbyint : fpc.Nearbyint,
    Round : fpc.Round,
    Trunc : fpc.Trunc,
    Acos : fpc.Acos,
    Asin : fpc.Asin,
    Atan : fpc.Atan,
    Cos : fpc.Cos,
    Sin : fpc.Sin,
    Tan : fpc.Tan,
    Acosh : fpc.Acosh,
    Asinh : fpc.Asinh,
    Atanh : fpc.Atanh,
    Cosh : fpc.Cosh,
    Sinh : fpc.Cosh,
    Tanh : fpc.Tanh,
    Exp : fpc.Exp,
    Exp2 : fpc.Exp2,
    Expm1 : fpc.Expm1,
    Log : fpc.Log,
    Log10 : fpc.Log10,
    Log1p : fpc.Log1p,
    Log2 : fpc.Log2,
    Erf : fpc.Erf,
    Erfc : fpc.Erfc,
    Lgamma : fpc.Lgamma,
    Tgamma : fpc.Tgamma,
    IsFinite : fpc.Isfinite,
    IsInf : fpc.Isinf,
    IsNan : fpc.Isnan,
    IsNormal : fpc.Isnormal,
    Signbit : fpc.Signbit,
    Not : fpc.Not,
    # binary
    Add : fpc.Add,
    Sub : fpc.Sub,
    Mul : fpc.Mul,
    Div : fpc.Div,
    Copysign : fpc.Copysign,
    Fdim : fpc.Fdim,
    Fmax : fpc.Fmax,
    Fmin : fpc.Fmin,
    Fmod : fpc.Fmod,
    Remainder : fpc.Remainder,
    Hypot : fpc.Hypot,
    Atan2 : fpc.Atan2,
    Pow : fpc.Pow,
    # ternary
    Fma : fpc.Fma,
    # N-ary
    Or : fpc.Or,
    And : fpc.And
}
