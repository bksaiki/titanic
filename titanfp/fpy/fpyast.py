"""AST nodes in the FPY language"""

from abc import ABC
from enum import Enum
from typing import Any, Optional, Self

class Ast(ABC):
    """Abstract base class for FPy AST nodes."""

    def __repr__(self):
        name = self.__class__.__name__
        items = ', '.join(f'{k}={repr(v)}' for k, v in self.__dict__.items())
        return f'{name}({items})'

class TypeAnn(Ast):
    """FPy node: type annotation"""

class AnyType(TypeAnn):
    """FPy node: type annotation"""

class RealType(TypeAnn):
    """FPy node: real type"""

class BoolType(TypeAnn):
    """FPy node: real type"""

class Argument(Ast):
    """FPy node: argument"""
    name: str
    ty: TypeAnn

    def __init__(self, name: str, ty: TypeAnn):
        self.name = name
        self.ty = ty

class Expr(Ast):
    """FPy node: abstract expression"""

class ValueExpr(Ast):
    """FPy node: abstract terminal"""

class Var(ValueExpr):
    """FPy node: variable"""
    name: str

    def __init__(self, name: str):
        self.name = name

class Num(ValueExpr):
    """FPy node: numerical constant"""
    val: str | int | float

    def __init__(self, val: str | int | float):
        self.val = val

class Integer(Num):
    """FPy node: numerical constant (integer)"""
    val: int

    def __init__(self, val: int):
        super().__init__(val)

class Digits(ValueExpr):
    """FPy node: numerical constant in scientific notation"""
    m: int
    e: int
    b: int

    def __init__(self, m: int, e: int, b: int):
        self.m = m
        self.e = e
        self.b = b

class IfExpr(Expr):
    """FPy node: if expression (ternary)"""
    cond: Expr
    ift: Expr
    iff: Expr

    def __init__(self, cond: Expr, ift: Expr, iff: Expr):
        self.cond = cond
        self.ift = ift
        self.iff = iff

class NaryExpr(Expr):
    """FPy node: application expression"""
    name: str
    children: list[Expr]

    def __init__(self, name: str, *children: Expr) -> None:
        self.name = name
        self.children = list(children)

class UnknownCall(NaryExpr):
    """FPy node: abstract application"""

class UnaryExpr(NaryExpr):
    """FPy node: abstract unary application"""

class Neg(UnaryExpr):
    """FPy node: subtraction expression"""
    
    def __init__(self, e: Expr):
        super().__init__('-', e)

class Fabs(UnaryExpr):
    """FPy node: absolute value expression"""
    
    def __init__(self, e: Expr):
        super().__init__('fabs', e)

class Sqrt(UnaryExpr):
    """FPy node: square-root expression"""
    
    def __init__(self, e: Expr):
        super().__init__('sqrt', e)

class BinaryExpr(NaryExpr):
    """FPy node: abstract ternary application"""

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

class Sin(UnaryExpr):
    """FPy node: sine expression"""

    def __init__(self, e: Expr):
        super().__init__('sin', e)

class Cos(UnaryExpr):
    """FPy node: cosine expression"""

    def __init__(self, e: Expr):
        super().__init__('cos', e)

class Tan(UnaryExpr):
    """FPy node: tangent expression"""

    def __init__(self, e: Expr):
        super().__init__('tan', e)

class Atan(UnaryExpr):
    """FPy node: inverse tangent expression"""

    def __init__(self, e: Expr):
        super().__init__('atan', e)

class Or(NaryExpr):
    """FPy node: || expression"""
    def __init__(self, *children: Expr):
        super().__init__('or', *children)

class And(NaryExpr):
    """FPy node: && expression"""
    def __init__(self, *children: Expr):
        super().__init__('and', *children)

class CompareOp(Enum):
    LT = 0
    LE = 1
    GE = 2
    GT = 3
    EQ = 4
    NE = 5

class Compare(Expr):
    """FPy node: N-argument comparison (N >= 2)"""
    ops: list[CompareOp]
    children: list[Expr]
        
    def __init__(self, ops: list[CompareOp], children: list[Expr]):
        if not isinstance(children, list) or len(children) < 2:
            raise TypeError('expected list of length >= 2', children)
        if not isinstance(ops, list) or len(ops) != len(children) - 1:
            raise TypeError(f'expected list of length >= {len(children)}', children)
        self.ops = ops
        self.children = children

class Stmt(Ast):
    """FPy node: abstract statement"""

class Block(Stmt):
    """FPy node: list of statements"""
    stmts: list[Stmt]

    def __init__(self, stmts: list[Stmt]):
        self.stmts = stmts

class Assign(Stmt):
    """FPy node: assignment"""
    name: str
    val: Expr
    ann: Optional[TypeAnn]

    def __init__(self, name: str, val: Expr, ann: Optional[TypeAnn] = None):
        self.name = name
        self.val = val
        self.ann = ann

class IfStmt(Stmt):
    """FPy node: if statement"""
    cond: Expr
    ift: Stmt
    iff: Stmt

    def __init__(self, cond: Expr, ift: Stmt, iff: Stmt):
        self.cond = cond
        self.ift = ift
        self.iff = iff

class Return(Stmt):
    """FPy node: return statement"""
    e: Expr

    def __init__(self, e: Expr):
        self.e = e

class Context(Ast):
    """FPy node: rounding contexts"""
    props: dict[str, Any]

    def __init__(self, props: dict[str, Any] = {}):
        self.props = props


class Function(Ast):
    """FPy node: function"""
    ident: Optional[str]
    args: list[Argument]
    body: Stmt
    ctx: Context

    name: Optional[str]
    pre: Optional[Self]

    def __init__(
        self,
        args: list[Argument],
        body: Stmt,
        ctx: Context = Context(),
        ident: Optional[str] = None,
        name: Optional[str] = None,
        pre: Optional[Self] = None,
    ):
        self.ident = ident
        self.args = args
        self.body = body
        self.ctx = ctx
        self.name = name
        self.pre = pre
