"""
Free variable analysis.

Compute the set of free variables of an FPy program.
"""

from ..fpyast import *
from ..visitor import Analysis

class FreeVars(Analysis):
    """
    Free variable analyzer.

    Compute the set of free variables of any FPy `Function`, `Stmt`, or `Expr`.
    """

    _ResultType = set[str]
    """
    Type of the `ctx` argument to each visitor method.

    The argument is the set of bound variables in the current scope.
    """

    def __init__(self, record=True):
        super().__init__('free_vars', record)

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

    def _visit_assign(self, stmt, ctx) -> _ResultType:
        raise NotImplementedError

    def _visit_tuple_assign(self, stmt, ctx) -> _ResultType:
        raise NotImplementedError

    def _visit_return(self, stmt, ctx) -> _ResultType:
        raise NotImplementedError

    def _visit_if_stmt(self, stmt, ctx) -> _ResultType:
        raise NotImplementedError

    def _visit_block(self, block, ctx) -> _ResultType:
        fvs: set[str] = set()
        for st in reversed(block.stmts):
            match st:
                case Assign():
                    fvs = fvs.difference(self._collect_vars(st.var))
                    fvs = fvs.union(self._visit(st.val, ctx))
                case TupleAssign():
                    fvs = fvs.difference(self._collect_vars(st.binding))
                    fvs = fvs.union(self._visit(st.val, ctx))
                case IfStmt():
                    cond_fvs = self._visit(st.cond, ctx)
                    ift_fvs = self._visit(st.ift, ctx)
                    iff_fvs = self._visit(st.iff, ctx)
                    return cond_fvs.union(ift_fvs, iff_fvs)
                case Return():
                    fvs = fvs.union(self._visit(st.e, ctx))
                case _:
                    raise NotImplementedError('unreachable', st)
        return fvs

    def _visit_function(self, func, ctx) -> _ResultType:
        fvs = self._visit_block(func.body, None)
        return fvs

    # override typing hint
    def _visit(self, e, ctx) -> _ResultType:
        return super()._visit(e, ctx)

    def visit(self, e: Function | Block | Stmt | Expr) -> _ResultType:
        if not (isinstance(e, Function) or isinstance(e, Block) \
                or isinstance(e, Stmt) or isinstance(e, Expr)):
            raise TypeError(f'visit() argument 1 must be Function, Stmt or Expr, not {e}')
        return self._visit(e, None)


    def _collect_vars(self, binding: Binding):
        """Returns the set of identifiers in a binding."""
        match binding:
            case VarBinding():
                return { binding.name }
            case TupleBinding():
                fvs: set[str] = set()
                for bind in binding.bindings:
                    fvs = fvs.union(self._collect_vars(bind))
                return fvs
            case _:
                raise NotImplementedError('unreachable', binding)

    

    # def _visit_assign(self, stmt, ctx: _CtxType):
    #     raise NotImplementedError('do not call directly')

    # def _visit_tuple_assign(self, stmt, ctx: _CtxType):
    #     raise NotImplementedError('do not call directly')

    # def _visit_return(self, stmt, ctx: _CtxType):
    #     raise NotImplementedError('do not call directly')

    # def _visit_if_stmt(self, stmt, ctx: _CtxType):
    #     raise NotImplementedError('do not call directly')
    #     # cond_fvs = self._visit(stmt.cond, ctx)
    #     # ift_ctx, ift_fvs = self._visit_block(stmt.ift, ctx)
    #     # iff_ctx, iff_fvs = self._visit_block(stmt.iff, ctx)
    #     # return cond_fvs.union(ift_fvs, iff_fvs)

    # def _visit_binding(self, binding: Binding, ctx: _CtxType) -> set[str]:
    #     match binding:
    #         case VarBinding():
    #             return { *ctx, binding.name }
    #         case TupleBinding():
    #             for elt in binding.bindings:
    #                 ctx = self._visit_binding(elt, ctx)
    #             return ctx
    #         case _:
    #             raise NotImplementedError('unreachable', binding)

    # def _visit_block(self, block, ctx: _CtxType) -> tuple[_CtxType, set[str]]:
    #     fvs: set[str] = set()
    #     for st in reversed(block.stmts):
    #         match st:
    #             case Assign():
    #                 fvs = fvs.union(self._visit(st.val, ctx))
    #                 ctx = self._visit_binding(st.var, ctx)
    #             # case TupleAssign():
    #             #     fvs = fvs.union(self._visit(st.val, ctx))
    #             #     ctx = self._visit_binding(st.binding, ctx)
    #             case Return():
    #                 fvs = fvs.union(self._visit(st.e, ctx))
    #             # case IfStmt():
    #             #     cond_fvs = self._visit(st.cond, ctx)
    #             #     ift_ctx, ift_fvs = self._visit_block(st.ift, ctx)
    #             #     iff_ctx, iff_fvs = self._visit_block(st.iff, ctx)
    #             #     fvs = fvs.union(cond_fvs, ift_fvs, iff_fvs)
    #             #     ctx = ift_ctx.intersection(iff_ctx)
    #             case _:
    #                 raise NotImplementedError('unreachable', st)
    #     return ctx, fvs

    # def _visit_function(self, func, ctx: _CtxType):
    #     new_ctx: set[str] = set()
    #     for arg in func.args:
    #         new_ctx.add(arg.name)
    #     _, fvs = self._visit_block(func.body, new_ctx)
    #     return fvs

    # # override typing hint
    # def _visit(self, e, ctx: _CtxType) -> set[str]:
    #     return super()._visit(e, ctx)

    # def visit(self, e: Function | Stmt | Expr):
    

