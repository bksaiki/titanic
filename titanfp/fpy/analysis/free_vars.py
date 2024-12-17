"""
Free variable analysis.

Compute the set of free variables of an FPy program.
"""

from ..fpyast import *
from ..visitor import ReduceVisitor

class FreeVars(ReduceVisitor):
    """
    Free variable analyzer.

    Compute the set of free variables of any FPy `Function`, `Stmt`, or `Expr`.
    """

    _CtxType = set[str]
    """
    Type of the `ctx` argument to each visitor method.

    The argument is the set of bound variables in the current scope.
    """

    def _visit_decnum(self, e, ctx: _CtxType):
        return set()

    def _visit_integer(self, e, ctx: _CtxType):
        return set()

    def _visit_digits(self, e, ctx: _CtxType):
        return set()

    def _visit_variable(self, e, ctx: _CtxType):
        if e.name in ctx:
            return set()
        else:
            return { e.name }

    def _visit_array(self, e, ctx: _CtxType):
        fvs: set[str] = set()
        for c in e.children:
            fvs = fvs.union(self._visit(c, ctx))
        return fvs

    def _visit_unknown(self, e, ctx: _CtxType):
        fvs: set[str] = set()
        for c in e.children:
            fvs = fvs.union(self._visit(c, ctx))
        return fvs

    def _visit_nary_expr(self, e, ctx: _CtxType):
        fvs: set[str] = set()
        for c in e.children:
            fvs = fvs.union(self._visit(c, ctx))
        return fvs

    def _visit_compare(self, e, ctx: _CtxType):
        fvs: set[str] = set()
        for c in e.children:
            fvs = fvs.union(self._visit(c, ctx))
        return fvs

    def _visit_if_expr(self, e, ctx: _CtxType):
        cond_fvs = self._visit(e.cond, ctx)
        ift_fvs = self._visit(e.ift, ctx)
        iff_fvs = self._visit(e.iff, ctx)
        return cond_fvs.union(ift_fvs, iff_fvs)

    def _visit_assign(self, stmt, ctx: _CtxType):
        return self._visit(stmt.val, ctx)

    def _visit_tuple_assign(self, stmt, ctx: _CtxType):
        return self._visit(stmt.val, ctx)

    def _visit_return(self, stmt, ctx: _CtxType):
        return self._visit(stmt.e, ctx)

    def _visit_if_stmt(self, stmt, ctx: _CtxType):
        cond_fvs = self._visit(stmt.cond, ctx)
        ift_fvs = self._visit(stmt.ift, ctx)
        iff_fvs = self._visit(stmt.iff, ctx)
        return cond_fvs.union(ift_fvs, iff_fvs)

    def _visit_binding(self, binding: Binding, ctx: _CtxType) -> set[str]:
        match binding:
            case VarBinding():
                return { *ctx, binding.name }
            case TupleBinding():
                for elt in binding.bindings:
                    ctx = self._visit_binding(elt, ctx)
                return ctx
            case _:
                raise NotImplementedError('unreachable', binding)

    def _visit_block(self, stmt, ctx: _CtxType):
        fvs: set[str] = set()
        for st in stmt.stmts:
            match st:
                case Assign():
                    fvs = fvs.union(self._visit(st, ctx))
                    ctx = self._visit_binding(st.var, ctx)
                case TupleAssign():
                    fvs = fvs.union(self._visit(st, ctx))
                    ctx = self._visit_binding(st.binding, ctx)
                case Return():
                    fvs = fvs.union(self._visit(st, ctx))
                case IfStmt():
                    fvs = fvs.union(self._visit(st, ctx))
                case _:
                    raise NotImplementedError('unreachable', st)

        return fvs

    def _visit_function(self, func, ctx: _CtxType):
        new_ctx: set[str] = set()
        for arg in func.args:
            new_ctx.add(arg.name)
        return self._visit(func.body, new_ctx)

    # override typing hint
    def _visit(self, e, ctx: _CtxType) -> set[str]:
        return super()._visit(e, ctx)

    def visit(self, e: Function | Stmt | Expr):
        if not (isinstance(e, Function) or isinstance(e, Stmt) or isinstance(e, Expr)):
            raise TypeError(f'visit() argument 1 must be Function, Stmt or Expr, not {e}')
        return self._visit(e, set())

