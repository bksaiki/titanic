"""Live variable analysis for the FPy AST."""

from .fpyast import *
from .visitor import AstVisitor

_LiveSet = set[str]

class LiveVars(AstVisitor):
    """Live variable analyzer on the AST."""

    analysis_name = 'live_vars'

    def analyze(self, func: Function):
        """Analyze the live variables in a function."""
        if not isinstance(func, Function):
            raise TypeError(f'LiveVars: expected a Function, got {func}')
        self._visit(func, set())

    def _visit_var(self, e, ctx) -> _LiveSet:
        return { e.name}

    def _visit_decnum(self, e, ctx) -> _LiveSet:
        return set()

    def _visit_integer(self, e, ctx) -> _LiveSet:
        return set()

    def _visit_unaryop(self, e, ctx) -> _LiveSet:
        return self._visit(e.arg, ctx)

    def _visit_binaryop(self, e, ctx) -> _LiveSet:
        return self._visit(e.left, ctx) | self._visit(e.right, ctx)

    def _visit_ternaryop(self, e, ctx) -> _LiveSet:
        live0 = self._visit(e.arg0, ctx)
        live1 = self._visit(e.arg1, ctx)
        live2 = self._visit(e.arg2, ctx)
        return live0 | live1 | live2

    def _visit_naryop(self, e, ctx) -> _LiveSet:
        live: set[str] = set()
        for arg in e.args:
            live |= self._visit(arg, ctx)
        return live

    def _visit_compare(self, e, ctx) -> _LiveSet:
        live: set[str] = set()
        for arg in e.args:
            live |= self._visit(arg, ctx)
        return live

    def _visit_call(self, e, ctx) -> _LiveSet:
        live: set[str] = set()
        for arg in e.args:
            live |= self._visit(arg, ctx)
        return live

    def _visit_tuple_expr(self, e, ctx) -> _LiveSet:
        live: set[str] = set()
        for arg in e.args:
            live |= self._visit(arg, ctx)
        return live

    def _visit_if_expr(self, e, ctx) -> _LiveSet:
        cond_live = self._visit(e.cond, ctx)
        ift_live = self._visit(e.ift, ctx)
        iff_live = self._visit(e.iff, ctx)
        return cond_live | ift_live | iff_live

    def _visit_var_assign(self, stmt, ctx) -> _LiveSet:
        raise NotImplementedError('do not call directly')

    def _visit_tuple_assign(self, stmt, ctx) -> _LiveSet:
        raise NotImplementedError('do not call directly')

    def _visit_if_stmt(self, stmt, ctx) -> _LiveSet:
        raise NotImplementedError('do not call directly')

    def _visit_while_stmt(self, stmt, ctx) -> _LiveSet:
        raise NotImplementedError('do not call directly')

    def _visit_for_stmt(self, stmt, ctx) -> _LiveSet:
        raise NotImplementedError('do not call directly')

    def _visit_return(self, stmt, ctx) -> _LiveSet:
        raise NotImplementedError('do not call directly')

    def _visit_block(self, block, ctx: _LiveSet) -> _LiveSet:
        block_out_live = set(ctx)
        live = set(block_out_live)
        for i, stmt in enumerate(reversed(block.stmts)):
            out_live = set(live)
            match stmt:
                case VarAssign():
                    live -= {stmt.var}
                    live |= self._visit(stmt.expr, None)
                case TupleAssign():
                    live -= stmt.vars.names()
                    live |= self._visit(stmt.expr, None)
                case IfStmt():
                    if stmt.iff is not None:
                        live |= self._visit(stmt.iff, ctx)
                    live |= self._visit(stmt.ift, ctx)
                    live |= self._visit(stmt.cond, None)
                case WhileStmt():
                    live |= self._visit(stmt.body, ctx)
                    live |= self._visit(stmt.cond, None)
                case ForStmt():
                    live |= self._visit(stmt.body, ctx)
                    live -= {stmt.var}
                    live |= self._visit(stmt.iterable, None)
                case Return():
                    assert i == 0, 'return statement must be the last statement'
                    out_live = set() # override incoming out set
                    live = self._visit(stmt.expr, None)
                case _:
                    raise NotImplementedError('unexpected statement', stmt)
            stmt.attribs[self.analysis_name] = (set(live), out_live)
        
        stmt.attribs[self.analysis_name] = (set(live), block_out_live)
        return live

    def _visit_function(self, func, ctx: _LiveSet):
        self._visit(func.body, ctx)
