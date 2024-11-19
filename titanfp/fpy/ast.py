"""AST nodes in the FPY language"""

from abc import ABC
from dataclasses import dataclass
from typing import Any, Callable

from ..titanic.digital import Digital

from .utils import raise_type_error

class Expr(ABC):
    """Abstract base class for FPy AST nodes."""
    # def __add__(self, other):
    #     return Add(self, other)
    
    # def __sub__(self, other):
    #     return Sub(self, other)
    
    # def __mul__(self, other):
    #     return Mul(self, other)
    
    # def __div__(self, other):
    #     return Div(self, other)

@dataclass
class Real(Expr):
    """FPy node: numerical constant"""
    val: int | float | Digital

# class ControlExpr()

# _Bool = bool | str | Ast
# _Real = int | float | Digital | str | Ast
# _Scalar = _Bool | _Real

# @dataclass
# class Bool(Ast):
#     """FPY node: boolean constant"""
#     val: _Bool

# @dataclass
# class Real(Ast):
#     """FPY node: numerical constant"""
#     val: _Real

# @dataclass
# class Add(Ast):
#     lhs: Ast
#     rhs: Ast

# @dataclass
# class Sub(Ast):
#     lhs: Ast
#     rhs: Ast

# @dataclass
# class Mul(Ast):
#     lhs: Ast
#     rhs: Ast

# @dataclass
# class Div(Ast):
#     lhs: Ast
#     rhs: Ast

# @dataclass
# class Sqrt(Ast):
#     arg: Ast

# @dataclass
# class LessEqual(Ast):
#     lhs: Ast
#     rhs: Ast

# @dataclass
# class If(Ast):
#     cond: Ast
#     ift: Ast
#     iff: Ast

class FPCore(Expr):
    """FPY node: function"""
    func: Callable[..., Any]
    arg_types: dict[str, type]
    source: str

    def __init__(
        self,
        func: Callable[..., Any],
        arg_types: dict[str, type],
        source: str
    ):
        self.func = func
        self.arg_types = arg_types
        self.source = source

    def __repr__(self):
        return f'FPCore(func={self.func}, ...)'

    def __call__(self, *args):
        if len(args) != len(self.arg_types):
            raise RuntimeError(f'arity mismtach, expected {len(self.arg_types)}, got {len(args)}', args)

        for arg, (name, ty) in zip(args, self.arg_types.items()):
            if not isinstance(arg, ty):
                raise_type_error(ty, arg)

        return self.func(*args)
        