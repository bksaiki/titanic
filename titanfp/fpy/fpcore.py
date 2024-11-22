"""Compilation from FPy to FPCore"""

from typing import Callable, Type

from .fpyast import *
from ..fpbench import fpcast as fpc

def _compile_argument(arg: Argument):
    match arg.ty:
        case RealType():
            return arg.name, None, None
        case _:
            raise NotImplementedError(arg)
        
_unary_table : dict[Type[NaryExpr], Callable[..., fpc.Expr]] = {
    Neg : fpc.Neg,
    Fabs : fpc.Fabs,
    Sqrt : fpc.Sqrt,
    Sin : fpc.Sin,
    Cos : fpc.Cos,
    Tan : fpc.Tan,
    Atan : fpc.Atan
}

_binary_table : dict[Type[NaryExpr], Callable[..., fpc.Expr]] = {
    Add : fpc.Add,
    Sub : fpc.Sub,
    Mul : fpc.Mul,
    Div : fpc.Div,
}

def _compile_expr(e: Expr):
    match e:
        case UnknownCall(name=name, children=children):
            return fpc.UnknownOperator(name=name, *map(_compile_expr, children))
        case NaryExpr(name=name, children=children):
            cls = _unary_table.get(type(e), None)
            if cls is not None:
                return cls(_compile_expr(children[0]))
            
            cls = _binary_table.get(type(e), None)
            if cls is not None:
                return cls(_compile_expr(children[0]), _compile_expr(children[1]))

            raise NotImplementedError(e)
        case Var(name=name):
            return fpc.Var(name)
        case Num(val=val):
            if isinstance(val, int):
                return fpc.Integer(val)
            elif isinstance(val, float):
                return fpc.Decnum(str(val))
            else:
                raise NotImplementedError(e)
        case Digits(m=m, e=exp, b=base):
            return fpc.Digits(m, exp, base)
        case _:
            raise NotImplementedError(e)

def _compile_block(block: Block):
   match block.stmts:
        case [stmt]:
            return _compile_statement(stmt)
        case [stmt, *stmts]:
            match stmt:
                case Assign(name=name, val=val):
                    return fpc.Let([(name, _compile_expr(val))], _compile_block(Block(stmts)))
                case _:
                    raise NotImplementedError('unexpected statement', stmt)
        case _:
           raise NotImplementedError('unexpected block', block.stmts)

def _compile_statement(stmt: Stmt):
    match stmt:
        case Block():
            return _compile_block(stmt)
        case Return(e=expr):
            return _compile_expr(expr)
        case _:
            raise NotImplementedError(stmt)

def _compile_func(func: Function):
    args = list(map(_compile_argument, func.args))
    # TODO: parse data
    props = func.ctx.props
    e = _compile_statement(func.body)

    return fpc.FPCore(
        inputs=args,
        e=e,
        props=props,
        ident=func.ident,
        name=func.name,
        pre=func.pre
    )

def fpy_to_fpcore(func: Function):
    if not isinstance(func, Function):
        raise TypeError(f'expected Function: {func}')
    return _compile_func(func)
