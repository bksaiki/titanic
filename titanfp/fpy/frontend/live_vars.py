"""Live variable analysis for the FPy AST."""

from .fpyast import *
from .visitor import AstVisitor

_ReturnType = set[str]

class LiveVars(AstVisitor):
    """Live variable analyzer on the AST."""

    analysis_name = 'live_vars'

    def analyze(self, func: Function):
        """Analyze the live variables in a function."""
        if not isinstance(func, Function):
            raise TypeError(f'LiveVars: expected a Function, got {func}')
        self._visit(func, None)

    def _visit_var(self, e, ctx) -> _ReturnType:
        return { e.name}

    def _visit_decnum(self, e, ctx) -> _ReturnType:
        return set()

    def _visit_integer(self, e, ctx) -> _ReturnType:
        return set()

    def _visit_unaryop(self, e, ctx) -> _ReturnType:
        return self._visit(e.arg, ctx)

    def _visit_binaryop(self, e, ctx) -> _ReturnType:
        return self._visit(e.left, ctx) | self._visit(e.right, ctx)

    def _visit_ternaryop(self, e, ctx) -> _ReturnType:
        fvs1 = self._visit(e.arg1, ctx)
        fvs2 = self._visit(e.arg2, ctx)
        fvs3 = self._visit(e.arg3, ctx)
        return fvs1 | fvs2 | fvs3

    def _visit_naryop(self, e, ctx) -> _ReturnType:
        fvs: set[str] = set()
        for arg in e.args:
            fvs |= self._visit(arg, ctx)
        return fvs

    def _visit_compare(self, e, ctx) -> _ReturnType:
        fvs: set[str] = set()
        for arg in e.args:
            fvs |= self._visit(arg, ctx)
        return fvs

    def _visit_call(self, e, ctx) -> _ReturnType:
        fvs: set[str] = set()
        for arg in e.args:
            fvs |= self._visit(arg, ctx)
        return fvs

    def _visit_tuple_expr(self, e, ctx) -> _ReturnType:
        fvs: set[str] = set()
        for arg in e.args:
            fvs |= self._visit(arg, ctx)
        return fvs

    def _visit_if_expr(self, e, ctx) -> _ReturnType:
        fvs1 = self._visit(e.cond, ctx)
        fvs2 = self._visit(e.ift, ctx)
        fvs3 = self._visit(e.iff, ctx)
        return fvs1 | fvs2 | fvs3

    def _visit_var_assign(self, stmt, ctx) -> _ReturnType:
        raise NotImplementedError('do not call directly')

    def _visit_tuple_assign(self, stmt, ctx) -> _ReturnType:
        raise NotImplementedError('do not call directly')

    def _visit_if_stmt(self, stmt, ctx) -> _ReturnType:
        raise NotImplementedError('do not call directly')

    def _visit_while_stmt(self, stmt, ctx) -> _ReturnType:
        raise NotImplementedError('do not call directly')

    def _visit_for_stmt(self, stmt, ctx) -> _ReturnType:
        raise NotImplementedError('do not call directly')

    def _visit_return(self, stmt, ctx) -> _ReturnType:
        raise NotImplementedError('do not call directly')

    def _visit_block(self, block, ctx) -> _ReturnType:
        fvs: set[str] = set()
        for i, stmt in enumerate(reversed(block.stmts)):
            out_fvs = set(fvs)
            match stmt:
                case VarAssign():
                    fvs -= {stmt.var}
                    fvs |= self._visit(stmt.expr, ctx)
                case TupleAssign():
                    fvs -= stmt.vars.names()
                    fvs |= self._visit(stmt.expr, ctx)
                case IfStmt():
                    if stmt.iff is not None:
                        fvs |= self._visit(stmt.iff, ctx)
                    fvs |= self._visit(stmt.ift, ctx)
                    fvs |= self._visit(stmt.cond, ctx)
                case WhileStmt():
                    fvs |= self._visit(stmt.body, ctx)
                    fvs |= self._visit(stmt.cond, ctx)
                case Return():
                    assert i == 0, 'return statement must be the last statement'
                    fvs = self._visit(stmt.expr, ctx)
                case _:
                    raise NotImplementedError('unexpected statement', stmt)
            stmt.attribs[self.analysis_name] = (set(fvs), out_fvs)
        return fvs

    def _visit_function(self, func, ctx):
        self._visit(func.body, ctx)
