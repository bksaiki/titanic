"""
Transformation pass to bundle updated variables in while loops
into a single variable.
"""

from typing import Optional

from .define_use import DefineUse
from .verify import VerifyIR
from ..ir import *
from ..utils import Gensym

_CtxType = dict[str, Expr]

class _WhileBundlingInstance(DefaultTransformVisitor):
    """Single-use instance of the WhileBundling pass."""
    func: Function
    gensym: Gensym

    def __init__(self, func: Function, names: set[str]):
        self.func = func
        self.gensym = Gensym(*names)

    def apply(self) -> Function:
        return self._visit(self.func, {})
    
    def _visit_var(self, e: Var, ctx: _CtxType):
        if e.name in ctx:
            return ctx[e.name]
        else:
            return Var(e.name)

    def _visit_while_stmt(self, stmt: WhileStmt, ctx: _CtxType):
        if len(stmt.phis) <= 1:
            return super()._visit_while_stmt(stmt, set(ctx))
        else:
            # unified phi variable
            phi_init = self.gensym.fresh('t')
            phi_name = self.gensym.fresh('t')
            phi_update = self.gensym.fresh('t')
            phi_node = PhiNode(phi_name, phi_init, phi_update, AnyType())
            # decompose current phi variables
            phi_names = [phi.name for phi in stmt.phis]
            phi_inits = [Var(phi.lhs) for phi in stmt.phis]
            phi_updates = [Var(phi.rhs) for phi in stmt.phis]
            # context for loop condition and body
            body_ctx = ctx.copy()
            for i, phi in enumerate(stmt.phis):
                body_ctx[phi.name] = RefExpr(Var(phi_name), Integer(i))
            # compile condition and body
            cond: Expr = self._visit(stmt.cond, body_ctx)
            body: Block = self._visit(stmt.body, body_ctx)
            # construct the block
            init_stmt = VarAssign(phi_init, AnyType(), TupleExpr(*phi_inits))
            update_stmt = VarAssign(phi_update, AnyType(), TupleExpr(*phi_updates))
            while_stmt = WhileStmt(cond, Block(body.stmts + [update_stmt]), [phi_node])
            unpack_stmt = TupleAssign(TupleBinding(phi_names), AnyType(), Var(phi_name))
            return Block([init_stmt, while_stmt, unpack_stmt])

    def _visit_block(self, block: Block, ctx: _CtxType):
        stmts: list[Stmt] = []
        for stmt in block.stmts:
            if isinstance(stmt, WhileStmt):
                stmt_or_block = self._visit_while_stmt(stmt, ctx)
                if isinstance(stmt_or_block, Stmt):
                    stmts.append(stmt_or_block)
                elif isinstance(stmt_or_block, Block):
                    stmts.extend(stmt_or_block.stmts)
                else:
                    raise NotImplementedError('unexpected', stmt_or_block)
            else:
                stmts.append(self._visit(stmt, ctx))
        return Block(stmts)

class WhileBundling:
    """
    Transformation pass to bundle updated variables in while loops.

    This pass rewrites the IR to bundle updated variables in while loops
    into a single variable. This transformation ensures there is only
    one phi node per while loop.
    """

    @staticmethod
    def apply(func: Function, names: Optional[set[str]] = None) -> Function:
        if names is None:
            uses = DefineUse.analyze(func)
            names = set(uses.keys())
        ir = _WhileBundlingInstance(func, names).apply()
        VerifyIR.check(ir)
        return ir
