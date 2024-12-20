"""
This module does intermediate code generation, compiling
the abstract syntax tree (AST) to the intermediate representation (IR).
"""

from typing import Any

from .fpyast import *
from .visitor import AstVisitor
from .. import ir
from ..gensym import Gensym

_CtxType = dict[str, str]

class _IRCodegenInstance(AstVisitor):
    """Instance of lowering an AST to an IR."""
    func: Function
    gensym: Gensym

    def __init__(self, func: Function):
        self.func = func
        self.gensym = Gensym()

    def lower(self) -> ir.Function:
        return self._visit(self.func, {})

    def _visit_var(self, e, ctx: _CtxType):
        return ir.Var(ctx[e.name])

    def _visit_decnum(self, e, ctx: _CtxType):
        return ir.Decnum(e.val)

    def _visit_integer(self, e, ctx: _CtxType):
        return ir.Integer(e.val)

    def _visit_unaryop(self, e, ctx: _CtxType):
        match e.op:
            case UnaryOpKind.NEG:
                arg = self._visit(e.arg, ctx)
                return ir.Neg(arg)
            case UnaryOpKind.NOT:
                arg = self._visit(e.arg, ctx)
                return ir.Not(arg)
            case _:
                raise NotImplementedError('unexpected op', e.op)

    def _visit_binaryop(self, e, ctx: _CtxType):
        lhs = self._visit(e.left, ctx)
        rhs = self._visit(e.right, ctx)
        match e.op:
            case BinaryOpKind.ADD:
                return ir.Add(lhs, rhs)
            case BinaryOpKind.SUB:
                return ir.Sub(lhs, rhs)
            case BinaryOpKind.MUL:
                return ir.Mul(lhs, rhs)
            case BinaryOpKind.DIV:
                return ir.Div(lhs, rhs)
            case _:
                raise NotImplementedError('unexpected op', e.op)

    def _visit_ternaryop(self, e, ctx: _CtxType):
        arg0 = self._visit(e.arg0, ctx)
        arg1 = self._visit(e.arg1, ctx)
        arg2 = self._visit(e.arg2, ctx)
        match e.op:
            case TernaryOpKind.FMA:
                return ir.Fma(arg0, arg1, arg2)
            case TernaryOpKind.DIGITS:
                assert isinstance(arg0, ir.Integer), f'must be an integer, got {arg0}'
                assert isinstance(arg1, ir.Integer), f'must be an integer, got {arg1}'
                assert isinstance(arg2, ir.Integer), f'must be an integer, got {arg2}'
                return ir.Digits(arg0.val, arg1.val, arg2.val)
            case _:
                raise NotImplementedError('unexpected op', e.op)

    def _visit_naryop(self, e, ctx: _CtxType):
        args: list[ir.Expr] = [self._visit(arg, ctx) for arg in e.args]
        match e.op:
            case NaryOpKind.AND:
                return ir.And(*args)
            case NaryOpKind.OR:
                return ir.Or(*args)
            case _:
                raise NotImplementedError('unexpected op', e.op)

    def _visit_compare(self, e, ctx: _CtxType):
        args: list[ir.Expr] = [self._visit(arg, ctx) for arg in e.args]
        

        raise NotImplementedError

    def _visit_call(self, e, ctx: _CtxType):
        args: list[ir.Expr]  = [self._visit(arg, ctx) for arg in e.args]
        return ir.UnknownCall(e.op, *args)

    def _visit_tuple_expr(self, e, ctx: _CtxType):
        return ir.TupleExpr(*[self._visit(arg, ctx) for arg in e.args])

    def _visit_if_expr(self, e, ctx: _CtxType):
        return ir.IfExpr(
            self._visit(e.cond, ctx),
            self._visit(e.ift, ctx),
            self._visit(e.iff, ctx)
        )

    def _visit_var_assign(self, stmt, ctx: _CtxType):
        t = self.gensym.fresh(stmt.var)
        ctx = { **ctx, stmt.var: t }
        e = self._visit(stmt.expr, ctx)
        s = ir.VarAssign(t, ir.AnyType(), e)
        return [s], ctx 

    def _visit_tuple_assign(self, stmt, ctx: _CtxType):
        raise NotImplementedError

    def _visit_if_stmt(self, stmt, ctx: _CtxType):
        raise NotImplementedError

    def _visit_while_stmt(self, stmt, ctx: _CtxType):
        raise NotImplementedError

    def _visit_for_stmt(self, stmt, ctx: _CtxType):
        raise NotImplementedError

    def _visit_return(self, stmt, ctx: _CtxType):
        e = self._visit(stmt.expr, ctx)
        s = ir.Return(e)
        return [s], set()

    def _visit_block(self, block, ctx: _CtxType):
        stmts: list[ir.Stmt] = []
        for stmt in block.stmts:
            match stmt:
                case VarAssign():
                    new_stms, ctx = self._visit(stmt, ctx)
                    stmts.extend(new_stms)
                case Return():
                    new_stms, ctx = self._visit(stmt, ctx)
                    stmts.extend(new_stms)
                case _:
                    raise NotImplementedError('unexpected statement', stmt)
        return ir.Block(stmts), ctx

    def _visit_function(self, func, ctx: _CtxType):
        ctx = dict(ctx)
        args: list[ir.Argument] = []
        for arg in func.args:
            ctx[arg.name] = self.gensym.reserve(arg)
            args.append(ir.Argument(arg.name, ir.AnyType()))
        e, _ = self._visit(func.body, ctx)
        return ir.Function(func.name, args, e, ir.AnyType()) 


class IRCodegen:
    """Lowers a FPy AST to FPy IR."""
    
    def lower(self, f: Function) -> ir.Function:
        return _IRCodegenInstance(f).lower()
