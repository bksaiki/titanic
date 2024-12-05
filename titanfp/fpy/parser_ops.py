"""Tables for the FPy parser."""

from typing import Callable

from .fpyast import *

unary_table: dict[str, Callable[..., Expr]] = {
    'fabs' : Fabs,
    'sqrt' : Sqrt,
    'cbrt' : Cbrt,
    'ceil' : Ceil,
    'floor' : Floor,
    'nearbyint' : Nearbyint,
    'round' : Round,
    'trunc' : Trunc,
    'acos' : Acos,
    'asin' : Asin,
    'atan' : Atan,
    'cos' : Cos,
    'sin' : Sin,
    'tan' : Tan,
    'acosh' : Acosh,
    'asinh' : Asinh,
    'atanh' : Atanh,
    'cosh' : Cosh,
    'sinh' : Sinh,
    'tanh' : Tanh,
    'exp' : Exp,
    'exp2' : Exp2,
    'expm1' : Expm1,
    'log' : Log,
    'log10' : Log10,
    'log1p' : Log1p,
    'log2' : Log2,
    'erf' : Erf,
    'erfc' : Erfc,
    'lgamma' : Lgamma,
    'tgamma' : Tgamma,
    'isfinite' : IsFinite,
    'isinf' : IsInf,
    'isnan' : IsNan,
    'isnormal' : IsNormal,
    'signbit' : Signbit,
    'not' : Not
}

binary_table: dict[str, Callable[..., Expr]] = {
    'copysign' : Copysign,
    'fdim' : Fdim,
    'max' : Fmax,
    'min' : Fmin,
    'fmod' : Fmod,
    'remainder' : Remainder,
    'hypot' : Hypot,
    'atan2' : Atan2,
    'pow' : Pow
}

ternary_table: dict[str, Callable[..., Expr]] = {
    'fma' : Fma
}
