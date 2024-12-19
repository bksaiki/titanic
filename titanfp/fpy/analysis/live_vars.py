"""
Free variable analysis.

Compute the set of free variables of an FPy program.
"""

from ..fpyast import *
from ..visitor import Analysis

class LiveVars(Analysis):
    """
    Free variable analyzer.

    Compute the set of free variables of any FPy `Function`, `Stmt`, or `Expr`.
    """

    _ResultType = set[str]
    """
    Type of the `ctx` argument to each visitor method.

    The argument is the set of bound variables in the current scope.
    """

    analysis_name = 'live_vars'
    """
    AST attribute key for this analysis.
    """

    def __init__(self, record = True):
        super().__init__(self.analysis_name, record)

    def _visit_decnum(self, e, ctx) -> _ResultType:
        return set()

    def _visit_integer(self, e, ctx) -> _ResultType:
        return set()

    def _visit_digits(self, e, ctx) -> _ResultType:
        return set()

    def _visit_variable(self, e, ctx) -> _ResultType:
        return { e.name }

    def _visit_array(self, e, ctx) -> _ResultType:
        fvs: set[str] = set()
        for c in e.children:
            fvs = fvs.union(self._visit(c, ctx))
        return fvs

    def _visit_unknown(self, e, ctx) -> _ResultType:
        fvs: set[str] = set()
        for c in e.children:
            fvs = fvs.union(self._visit(c, ctx))
        return fvs

    def _visit_nary_expr(self, e, ctx) -> _ResultType:
        fvs: set[str] = set()
        for c in e.children:
            fvs = fvs.union(self._visit(c, ctx))
        return fvs

    def _visit_compare(self, e, ctx) -> _ResultType:
        fvs: set[str] = set()
        for c in e.children:
            fvs = fvs.union(self._visit(c, ctx))
        return fvs

    def _visit_if_expr(self, e, ctx) -> _ResultType:
        cond_fvs = self._visit(e.cond, ctx)
        ift_fvs = self._visit(e.ift, ctx)
        iff_fvs = self._visit(e.iff, ctx)
        return cond_fvs.union(ift_fvs, iff_fvs)
    
    def _visit_assign(self, stmt, ctx):
        raise NotImplementedError('unreachable')
    
    def _visit_tuple_assign(self, stmt, ctx):
        raise NotImplementedError('unreachable')
    
    def _visit_return(self, stmt, ctx):
        raise NotImplementedError('unreachable')
    
    def _visit_if_stmt(self, stmt, ctx):
        raise NotImplementedError('unreachable')
    
    def _visit_while_stmt(self, stmt, ctx):
        raise NotImplementedError('unreachable')
    
    def _visit_phi(self, stmt, ctx):
        raise NotImplementedError('unreachable')

    def _visit_block(self, block, ctx: Optional[_ResultType]) -> _ResultType:
        # analysis runs in reverse, but visitor runs in forward order:
        # `ctx` is None if we are recursing downwards, else it is
        # the set of free variables from subsequence expressions
        if ctx is None:
            ctx = set()

        for stmt in reversed(block.stmts):
            out_ctx = ctx
            match stmt:
                case Assign():
                    ctx = ctx.difference(stmt.var.ids())
                    ctx = ctx.union(self._visit(stmt.val, None))
                    stmt.attribs[self.analysis_name] = (ctx, out_ctx)
                case TupleAssign():
                    ctx = ctx.difference(stmt.binding.ids())
                    ctx = ctx.union(self._visit(stmt.val, None))
                    stmt.attribs[self.analysis_name] = (ctx, out_ctx)
                case IfStmt():
                    cond_fvs = self._visit(stmt.cond, None)
                    ift_fvs = self._visit(stmt.ift, ctx)
                    if stmt.iff is None:
                        # 1-armed if
                        ctx = ctx.union(cond_fvs, ift_fvs)
                    else:
                        # 2-armed if
                        iff_fvs = self._visit(stmt.iff, ctx)
                        ctx = cond_fvs.union(ift_fvs, iff_fvs)
                    stmt.attribs[self.analysis_name] = (ctx, out_ctx)
                case WhileStmt():
                    cond_fvs = self._visit(stmt.cond, None)
                    body_fvs = self._visit(stmt.body, ctx)
                    ctx = cond_fvs.union(cond_fvs, body_fvs) # similar to a 1-armed if
                    stmt.attribs[self.analysis_name] = (ctx, out_ctx)
                case Return():
                    assert len(ctx) == 0, "return statement should be at the end of a block"
                    ctx = self._visit(stmt.e, None)
                    stmt.attribs[self.analysis_name] = (ctx, set())
                case Phi():
                    ctx = ctx.difference(stmt.name)
                    ctx = { *ctx, stmt.lhs, stmt.rhs }
                    stmt.attribs[self.analysis_name] = (ctx, out_ctx)
                case _:
                    raise NotImplementedError('unreachable', stmt)
        return ctx

    def _visit_function(self, func, ctx) -> _ResultType:
        return self._visit_block(func.body, ctx)

    # override typing hint
    def _visit(self, e, ctx) -> _ResultType:
        return super()._visit(e, ctx)

    def visit(self, e: Function | Block) -> _ResultType:
        if not (isinstance(e, Function | Block)):
            raise TypeError(f'visit() argument 1 must be Function or Block, not {e}')
        return self._visit(e, None)
