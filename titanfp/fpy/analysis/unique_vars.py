"""Computes the set of unique variables in an FPy program."""

from ..fpyast import *
from ..visitor import ReduceVisitor

class UniqueVars(ReduceVisitor):
    """
    Computes the set of unique variables in an FPy program.
    """

    _ResultType = set[str]

    def _visit_decnum(self, e, ctx) -> _ResultType:
        return set()

    def _visit_integer(self, e, ctx) -> _ResultType:
        return set()

    def _visit_digits(self, e, ctx) -> _ResultType:
        return set()

    def _visit_variable(self, e, ctx) -> _ResultType:
        return { e.name }

    def _visit_array(self, e, ctx) -> _ResultType:
        return set().union(*[self._visit(c, ctx) for c in e.children])

    def _visit_unknown(self, e, ctx) -> _ResultType:
        return set().union(*[self._visit(c, ctx) for c in e.children])

    def _visit_nary_expr(self, e, ctx) -> _ResultType:
        return set().union(*[self._visit(c, ctx) for c in e.children])

    def _visit_compare(self, e, ctx) -> _ResultType:
        return set().union(*[self._visit(c, ctx) for c in e.children])

    def _visit_if_expr(self, e, ctx) -> _ResultType:
        cond_vars = self._visit(e.cond, ctx)
        ift_vars = self._visit(e.ift, ctx)
        iff_vars = self._visit(e.iff, ctx)
        return cond_vars.union(ift_vars, iff_vars)

    def _visit_assign(self, stmt, ctx) -> _ResultType:
        return stmt.var.ids().union(self._visit(stmt.val, ctx))

    def _visit_tuple_assign(self, stmt, ctx) -> _ResultType:
        return stmt.binding.ids().union(self._visit(stmt.val, ctx))

    def _visit_return(self, stmt, ctx) -> _ResultType:
        return self._visit(stmt.e, ctx)

    def _visit_if_stmt(self, stmt, ctx) -> _ResultType:
        cond_vars = self._visit(stmt.cond, ctx)
        ift_vars = self._visit(stmt.ift, ctx)
        if stmt.iff is None:
            # 1-armed if
            return cond_vars.union(ift_vars)
        else:
            # 2-armed if
            iff_vars = self._visit(stmt.iff, ctx)
            return cond_vars.union(ift_vars, iff_vars)
        
    def _visit_while_stmt(self, stmt, ctx):
        cond_vars = self._visit(stmt.cond, ctx)
        body_vars = self._visit(stmt.body, ctx)
        return cond_vars.union(body_vars)

    def _visit_phi(self, stmt, ctx) -> _ResultType:
        return { stmt.name, stmt.lhs, stmt.rhs }

    def _visit_block(self, block, ctx) -> _ResultType:
        vars: set[str] = set()
        for stmt in block.stmts:
            vars = vars.union(self._visit(stmt, ctx))
        return vars

    def _visit_function(self, func, ctx) -> _ResultType:
        vars: set[str] = set()
        for arg in func.args:
            vars.add(arg.name)
        return vars.union(self._visit(func.body, None))

    # override to get typing hint
    def _visit(self, e, ctx) -> _ResultType:
        return super()._visit(e, ctx)

    def visit(self, e: Function | Block | Stmt | Expr) -> _ResultType:
        if not isinstance(e, Function | Block | Stmt | Expr):
            raise TypeError(f'visit() argument 1 must be Function, Block, Stmt, or Expr, not {e}')
        return self._visit(e, None)

    
