"""Live variable analysis for the FPy AST."""

from .fpyast import *
from .visitor import AstVisitor

_LiveSet = set[str]

class LiveVarAnalysis(AstVisitor):
    """Live variable analyzer on the AST."""

    analysis_name = 'live_vars'

    def analyze(self, func: Function):
        """Analyze the live variables in a function."""
        if not isinstance(func, Function):
            raise TypeError(f'expected a Function, got {func}')
        self._visit(func, set())

    def _visit_var(self, e, ctx) -> _LiveSet:
        live = { e.name }
        e.attribs[self.analysis_name] = set(live)
        return live

    def _visit_decnum(self, e, ctx) -> _LiveSet:
        e.attribs[self.analysis_name] = set()
        return set()

    def _visit_integer(self, e, ctx) -> _LiveSet:
        e.attribs[self.analysis_name] = set()
        return set()

    def _visit_unaryop(self, e, ctx) -> _LiveSet:
        live = self._visit(e.arg, ctx)
        e.attribs[self.analysis_name] = set(live)
        return live

    def _visit_binaryop(self, e, ctx) -> _LiveSet:
        live = self._visit(e.left, ctx) | self._visit(e.right, ctx)
        e.attribs[self.analysis_name] = set(live)
        return live

    def _visit_ternaryop(self, e, ctx) -> _LiveSet:
        live0 = self._visit(e.arg0, ctx)
        live1 = self._visit(e.arg1, ctx)
        live2 = self._visit(e.arg2, ctx)
        live = live0 | live1 | live2
        e.attribs[self.analysis_name] = set(live)
        return live

    def _visit_naryop(self, e, ctx) -> _LiveSet:
        live: set[str] = set()
        for arg in e.args:
            live |= self._visit(arg, ctx)
        e.attribs[self.analysis_name] = set(live)
        return live

    def _visit_compare(self, e, ctx) -> _LiveSet:
        live: set[str] = set()
        for arg in e.args:
            live |= self._visit(arg, ctx)
        e.attribs[self.analysis_name] = set(live)
        return live

    def _visit_call(self, e, ctx) -> _LiveSet:
        live: set[str] = set()
        for arg in e.args:
            live |= self._visit(arg, ctx)
        e.attribs[self.analysis_name] = set(live)
        return live

    def _visit_tuple_expr(self, e, ctx) -> _LiveSet:
        live: set[str] = set()
        for arg in e.args:
            live |= self._visit(arg, ctx)
        e.attribs[self.analysis_name] = set(live)
        return live

    def _visit_if_expr(self, e, ctx) -> _LiveSet:
        cond_live = self._visit(e.cond, ctx)
        ift_live = self._visit(e.ift, ctx)
        iff_live = self._visit(e.iff, ctx)
        live = cond_live | ift_live | iff_live
        e.attribs[self.analysis_name] = set(live)
        return live

    def _visit_var_assign(self, stmt, ctx: _LiveSet) -> _LiveSet:
        ctx -= {stmt.var}
        return ctx | self._visit(stmt.expr, None)

    def _visit_tuple_assign(self, stmt, ctx: _LiveSet) -> _LiveSet:
        ctx -= stmt.vars.names()
        return ctx | self._visit(stmt.expr, None)

    def _visit_if_stmt(self, stmt, ctx: _LiveSet) -> _LiveSet:
        if stmt.iff is None:
            ctx |= self._visit(stmt.ift, set(ctx))
        else:
            ift_ctx = self._visit(stmt.ift, set(ctx))
            iff_ctx = self._visit(stmt.iff, set(ctx))
            ctx = ift_ctx | iff_ctx
        return ctx | self._visit(stmt.cond, None)

    def _visit_while_stmt(self, stmt, ctx: _LiveSet) -> _LiveSet:
        ctx |= self._visit(stmt.body, set(ctx))
        return ctx | self._visit(stmt.cond, None)

    def _visit_for_stmt(self, stmt, ctx: _LiveSet) -> _LiveSet:
        ctx |= self._visit(stmt.body, set(ctx))
        ctx -= {stmt.var}
        return ctx | self._visit(stmt.iterable, None)

    def _visit_return(self, stmt, ctx: _LiveSet) -> _LiveSet:
        return self._visit(stmt.expr, None)

    def _visit_block(self, block, ctx: _LiveSet) -> _LiveSet:
        block_out = set(ctx)
        for i, stmt in enumerate(reversed(block.stmts)):
            if isinstance(stmt, Return):
                assert i == 0, 'return statement must be the last statement'
                stmt_out = set() # override incoming out set
            else:
                stmt_out = set(ctx)
            ctx = self._visit(stmt, ctx)
            stmt.attribs[self.analysis_name] = (set(ctx), stmt_out)

        block.attribs[self.analysis_name] = (set(ctx), block_out)
        return ctx

    def _visit_function(self, func, ctx: _LiveSet):
        self._visit(func.body, ctx)
