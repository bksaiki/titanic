"""
Single-Static Assignment (SSA) transformation.

This module provides functions to transform any FPy program into SSA form.
"""

from ..analysis.live_vars import LiveVars
from ..fpyast import *
from ..gensym import Gensym
from ..visitor import DefaultTransformVisitor

class _SSACtx:
        """
        Context for the SSA transformation.
        """
    
        gensym: Gensym
        """
        Unique name generator.
        """
        orig: dict[str, str]
        """
        Mapping from SSA name to the original variable name.
        """
        env: dict[str, str]
        """
        Mapping from variable names to their SSA names.
        """

        def __init__(self, x: Optional[Self] = None):
            if x is None:
                self.gensym = Gensym()
                self.orig = dict()
                self.env = dict()
            else:
                self.gensym = x.gensym
                self.orig = x.orig
                self.env = x.env.copy()

        def reserve(self, name: str):
            """
            Reserves a name in the generator. This mutates the current context.
            """
            self.gensym.reserve(name)
            self.env[name] = name
            self.orig[name] = name

        def fresh(self, var: str):
            """
            Extends the current context with a new variable and its SSA name.
            Returns the new context and the SSA name.
            """
            copy = _SSACtx(self)
            renamed = copy.gensym.fresh(var)
            copy.orig[renamed] = var
            copy.env[var] = renamed
            return copy, renamed

class SSA(DefaultTransformVisitor):
    """
    Visitor implementing the SSA transformation.

    A program in SSA form satisfies the following property:
    every variable is assigned once but used multiple times.
    Every updating assignment is replaced with an assignment
    to a new variable.
    For example,
    ```
    x = 1
    x = x + 1
    ```
    is replaced with
    ```
    x = 1
    x1 = x + 1
    ```
    where `x1` is a new variable.
    ```
    """

    def _visit_variable(self, e, ctx: _SSACtx):
        return Var(ctx.env[e.name])

    def _visit_assign(self, stmt, ctx):
        raise NotImplementedError('do not call directly')

    def _visit_tuple_assign(self, stmt, ctx: _SSACtx):
        raise NotImplementedError('do not call directly')

    def _visit_return(self, stmt, ctx: _SSACtx):
        raise NotImplementedError('do not call directly')

    def _visit_if_stmt(self, stmt, ctx: _SSACtx):
        raise NotImplementedError('do not call directly')

    def _visit_block(self, block, ctx: _SSACtx):
        stmts: list[Stmt] = []
        for i, stmt in enumerate(block.stmts):
            match stmt:
                case Assign():
                    ctx, renamed = self._rename_binding(stmt.var, ctx)
                    stmts.append(Assign(renamed, self._visit(stmt.val, ctx)))
                case TupleAssign():
                    ctx, renamed = self._rename_binding(stmt.binding, ctx)
                    stmts.append(TupleAssign(renamed, self._visit(stmt.val, ctx)))
                case Return():
                    stmts.append(Return(self._visit(stmt.e, ctx)))
                case IfStmt():
                    assert i + 1 < len(block.stmts), 'if statement must be followed by a statement'
                    # recurse on children
                    cond = self._visit(stmt.cond, ctx)
                    ift, ift_ctx = self._visit_block(stmt.ift, ctx)
                    iff, iff_ctx = self._visit_block(stmt.iff, ctx)
                    new_stmt = IfStmt(cond, ift, iff)
                    stmts.append(new_stmt)
                    # live variables after this statement
                    # need to merge them with Phi nodes
                    fvs = block.stmts[i + 1].attribs[LiveVars.analysis_name]
                    for fv in fvs:
                        assert fv in ift_ctx.env, f'variable {fv} not in ift_ctx'
                        assert fv in iff_ctx.env, f'variable {fv} not in iff_ctx'
                        ctx, renamed = ctx.fresh(fv)
                        stmts.append(Phi(renamed, ift_ctx.env[fv], iff_ctx.env[fv], new_stmt))
                case _:
                    return NotImplementedError('unreachable', stmt)
        return Block(stmts), ctx

    def _visit_function(self, func: Function, ctx: _SSACtx):
        for arg in func.args:
            ctx.reserve(arg.name)

        body, _ = self._visit(func.body, ctx)
        return Function(
            args=func.args,
            body=body,
            ctx=func.ctx,
            ident=func.ident,
            name=func.name,
            pre=func.pre,
        )

    def visit(self, e: Function | Block):
        if not isinstance(e, Function | Block):
            raise TypeError(f'visit() argument 1 must be Function or Block, not {e}')
        if LiveVars.analysis_name not in e.attribs:
            raise RuntimeError('must run LiveVars analysis to use this transformation')

        ctx = _SSACtx()
        new = self._visit(e, ctx)
        return (new, ctx.orig)

    def _rename_binding(self, binding: Binding, ctx: _SSACtx):
        match binding:
            case VarBinding():
                ctx, renamed = ctx.fresh(binding.name)
                return ctx, VarBinding(renamed)
            case TupleBinding():
                new_bindings: list[Binding] = []
                for b in binding.bindings:
                    ctx, renamed = self._rename_binding(b, ctx)
                    new_bindings.append(renamed)
                return ctx, TupleBinding(*new_bindings)
            case _:
                raise NotImplementedError('unexpected', binding)

