"""
This module lowers the abstract syntax tree (AST) to
the intermediate representation (IR).
"""

from typing import Any

from .fpyast import *
from .visitor import AstVisitor
from .. import ir

_CtxType = Any

class Lower(AstVisitor):
    """Lowers the AST to the IR."""

    func: Function

    def __init__(self, func: Function):
        self.func = func

    def lower(self):
        return self._visit(self.func, None)

    def _visit_var(self, e, ctx: _CtxType):
        raise NotImplementedError

    def _visit_decnum(self, e, ctx: _CtxType):
        raise NotImplementedError

    def _visit_integer(self, e, ctx: _CtxType):
        raise NotImplementedError

    def _visit_unaryop(self, e, ctx: _CtxType):
        raise NotImplementedError

    def _visit_binaryop(self, e, ctx: _CtxType):
        raise NotImplementedError

    def _visit_ternaryop(self, e, ctx: _CtxType):
        raise NotImplementedError

    def _visit_naryop(self, e, ctx: _CtxType):
        raise NotImplementedError

    def _visit_compare(self, e, ctx: _CtxType):
        raise NotImplementedError

    def _visit_call(self, e, ctx: _CtxType):
        raise NotImplementedError

    def _visit_tuple_expr(self, e, ctx: _CtxType):
        raise NotImplementedError

    def _visit_if_expr(self, e, ctx: _CtxType):
        raise NotImplementedError

    def _visit_var_assign(self, stmt, ctx: _CtxType):
        raise NotImplementedError

    def _visit_tuple_assign(self, stmt, ctx: _CtxType):
        raise NotImplementedError

    def _visit_if_stmt(self, stmt, ctx: _CtxType):
        raise NotImplementedError

    def _visit_while_stmt(self, stmt, ctx: _CtxType):
        raise NotImplementedError

    def _visit_for_stmt(self, stmt, ctx: _CtxType):
        raise NotImplementedError

    def _visit_return(self, stmt, ctx: _CtxType):
        raise NotImplementedError
    
    def _visit_block(self, block, ctx: _CtxType):
        raise NotImplementedError

    def _visit_function(self, func, ctx: _CtxType):
        args = [ir.Argument(arg.name, ir.AnyType()) for arg in func.args]
        e = self._visit(func.body, ctx)
        return ir.Function(func.name, args, e, ir.RealType())
