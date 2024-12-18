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

    def __init__(self, record = True):
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

    def _visit_assign(self, stmt, ctx: _ResultType) -> _ResultType:
        ctx = ctx.difference(self._collect_vars(stmt.var))
        return ctx.union(self._visit(stmt.val, None))

    def _visit_tuple_assign(self, stmt, ctx: _ResultType) -> _ResultType:
        ctx = ctx.difference(self._collect_vars(stmt.binding))
        return ctx.union(self._visit(stmt.val, None))

    def _visit_return(self, stmt, ctx: _ResultType) -> _ResultType:
        return ctx.union(self._visit(stmt.e, None))

    def _visit_if_stmt(self, stmt, ctx) -> _ResultType:
        ift_fvs = self._visit(stmt.ift, ctx)
        iff_fvs = self._visit(stmt.iff, ctx)
        cond_fvs = self._visit(stmt.cond, None)
        return cond_fvs.union(ift_fvs, iff_fvs)

    def _visit_block(self, block, ctx: Optional[_ResultType]) -> _ResultType:
        # analysis runs in reverse, but visitor runs in forward order:
        # `ctx` is None if we are recursing downwards, else it is
        # the set of free variables from subsequence expressions
        if ctx is None:
            ctx = set()

        for stmt in reversed(block.stmts):
            match stmt:
                case Assign():
                    ctx = self._visit(stmt, ctx)
                case TupleAssign():
                    ctx = self._visit(stmt, ctx)
                case IfStmt():
                    ctx = self._visit(stmt, ctx)
                case Return():
                    # no subsequent expression after a return statement
                    ctx = self._visit(stmt, set())
                case _:
                    raise NotImplementedError('unreachable', stmt)
        
        return ctx

        # fvs: set[str] = set()
        # for st in reversed(block.stmts):
        #     match st:
        #         case Assign():
        #             fvs = self._visit(st, fvs)
        #         case TupleAssign():
        #             fvs = self._visit(st, fvs)
        #         case IfStmt():
        #             cond_fvs = self._visit(st.cond, ctx)
        #             ift_fvs = self._visit(st.ift, ctx)
        #             iff_fvs = self._visit(st.iff, ctx)
        #             return cond_fvs.union(ift_fvs, iff_fvs)
        #         case Return():
        #             fvs = fvs.union(self._visit(st.e, ctx))
        #         case _:
        #             raise NotImplementedError('unreachable', st)
        # return fvs

    def _visit_function(self, func, ctx) -> _ResultType:
        return self._visit_block(func.body, ctx)

    # override typing hint
    def _visit(self, e, ctx) -> _ResultType:
        return super()._visit(e, ctx)

    def visit(self, e: Function | Block) -> _ResultType:
        if not (isinstance(e, Function) or isinstance(e, Block)):
            raise TypeError(f'visit() argument 1 must be Function or Block, not {e}')
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
