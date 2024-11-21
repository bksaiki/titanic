"""AST nodes in the FPY language"""

from abc import ABC
from typing import Optional

class Ast(ABC):
    """Abstract base class for FPy AST nodes."""

class TypeAnn(Ast):
    """FPy node: type annotation"""
    name: str

    def __init__(self, name: str):
        self.name = name

class AnyType(TypeAnn):
    """FPy node: type annotation"""

class RealType(TypeAnn):
    """FPy node: real type"""

class BoolType(TypeAnn):
    """FPy node: real type"""

class Argument(Ast):
    """FPy node: argument"""
    name: str

    def __init__(self, name: str):
        self.name = name

class Expr(Ast):
    """FPy node: abstract expression"""

class Real(Expr):
    """FPy node: numerical constant"""
    val: str | int | float

    def __init__(self, val: str | int | float):
        super().__init__()
        self.val = val

class NaryExpr(Expr):
    """FPy node: application expression"""
    name: str
    children: tuple[Expr, ...]

    def __init__(self, name: str, *children: Expr) -> None:
        self.name = name
        self.children = children

class UnknownCall(NaryExpr):
    """FPy node: abstract application"""

class UnaryExpr(NaryExpr):
    """FPy node: abstract unary application"""

    def __init__(self, name: str, e: Expr):
        super().__init__(name, e)

class Neg(UnaryExpr):
    """FPy node: subtraction expression"""
    
    def __init__(self, e: Expr):
        super().__init__('-', e)

class Sqrt(UnaryExpr):
    """FPy node: square-root expression"""
    
    def __init__(self, e: Expr):
        super().__init__('sqrt', e)

class BinaryExpr(NaryExpr):
    """FPy node: abstract ternary application"""
    
    def __init__(self, name: str, e1: Expr, e2: Expr):
        super().__init__(name, e1, e2)

class Add(BinaryExpr):
    """FPy node: subtraction expression"""
    
    def __init__(self, e1: Expr, e2: Expr):
        super().__init__('+', e1, e2)

class Sub(BinaryExpr):
    """FPy node: subtraction expression"""
    
    def __init__(self, e1: Expr, e2: Expr):
        super().__init__('-', e1, e2)

class Mul(BinaryExpr):
    """FPy node: subtraction expression"""

    def __init__(self, e1: Expr, e2: Expr):
        super().__init__('*', e1, e2)
    
class Div(BinaryExpr):
    """FPy node: subtraction expression"""

    def __init__(self, e1: Expr, e2: Expr):
        super().__init__('/', e1, e2)

class Stmt(Ast):
    """FPy node: abstract statement"""

class Assign(Stmt):
    """FPy node: assignment"""
    name: str
    val: Expr
    ann: Optional[TypeAnn]

    def __init__(self, name: str, val: Expr, ann: Optional[TypeAnn] = None):
        super().__init__()
        self.name = name
        self.val = val
        self.ann = ann

class Return(Stmt):
    """FPy node: return statement"""
    e: Expr

    def __init__(self, e: Expr):
        super().__init__()
        self.e = e

class Function(Ast):
    """FPy node: function"""
    name: str
    body: list[Stmt]

    def __init__(self, name: str, body: list[Stmt]):
        self.name = name
        self.body = body

# class Expr(ABC):
#     """Abstract base class for FPy AST nodes."""
#     # def __add__(self, other):
#     #     return Add(self, other)
    
#     # def __sub__(self, other):
#     #     return Sub(self, other)
    
#     # def __mul__(self, other):
#     #     return Mul(self, other)
    
#     # def __div__(self, other):
#     #     return Div(self, other)

# @dataclass
# class Real(Expr):
#     """FPy node: numerical constant"""
#     val: int | float | Digital

# # class ControlExpr()

# # _Bool = bool | str | Ast
# # _Real = int | float | Digital | str | Ast
# # _Scalar = _Bool | _Real

# # @dataclass
# # class Bool(Ast):
# #     """FPY node: boolean constant"""
# #     val: _Bool

# # @dataclass
# # class Real(Ast):
# #     """FPY node: numerical constant"""
# #     val: _Real

# # @dataclass
# # class Add(Ast):
# #     lhs: Ast
# #     rhs: Ast

# # @dataclass
# # class Sub(Ast):
# #     lhs: Ast
# #     rhs: Ast

# # @dataclass
# # class Mul(Ast):
# #     lhs: Ast
# #     rhs: Ast

# # @dataclass
# # class Div(Ast):
# #     lhs: Ast
# #     rhs: Ast

# # @dataclass
# # class Sqrt(Ast):
# #     arg: Ast

# # @dataclass
# # class LessEqual(Ast):
# #     lhs: Ast
# #     rhs: Ast

# # @dataclass
# # class If(Ast):
# #     cond: Ast
# #     ift: Ast
# #     iff: Ast

# class FPCore(Expr):
#     """FPY node: function"""
#     func: Callable[..., Any]
#     arg_types: dict[str, type]
#     source: str

#     def __init__(
#         self,
#         func: Callable[..., Any],
#         arg_types: dict[str, type],
#         source: str
#     ):
#         self.func = func
#         self.arg_types = arg_types
#         self.source = source

#     def __repr__(self):
#         return f'FPCore(func={self.func}, ...)'

#     def __call__(self, *args):
#         if len(args) != len(self.arg_types):
#             raise RuntimeError(f'arity mismtach, expected {len(self.arg_types)}, got {len(args)}', args)

#         for arg, (name, ty) in zip(args, self.arg_types.items()):
#             if not isinstance(arg, ty):
#                 raise_type_error(ty, arg)

#         return self.func(*args)
        