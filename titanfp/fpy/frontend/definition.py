"""
Definition analyzer for the FPy AST.
"""

from .fpyast import *
from .visitor import AstVisitor

_DefSet = set[str]

class DefinitionAnalysisInstance(AstVisitor):
    """Single-use definition analyzer."""
    func: Function

    def __init__(self, func: Function):
        self.func = func

    def analyze(self):
        """Analyze the function."""
        self._visit(self.func, set())

    def _visit_var(self, e, ctx):
        raise NotImplementedError('should not be called')

    def _visit_decnum(self, e, ctx):
        raise NotImplementedError('should not be called')

    def _visit_integer(self, e, ctx):
        raise NotImplementedError('should not be called')

    def _visit_unaryop(self, e, ctx):
        raise NotImplementedError('should not be called')

    def _visit_binaryop(self, e, ctx):
        raise NotImplementedError('should not be called')

    def _visit_ternaryop(self, e, ctx):
        raise NotImplementedError('should not be called')

    def _visit_naryop(self, e, ctx):
        raise NotImplementedError('should not be called')

    def _visit_compare(self, e, ctx):
        raise NotImplementedError('should not be called')

    def _visit_call(self, e, ctx):
        raise NotImplementedError('should not be called')

    def _visit_tuple_expr(self, e, ctx):
        raise NotImplementedError('should not be called')

    def _visit_comp_expr(self, e, ctx):
        raise NotImplementedError('should not be called')

    def _visit_if_expr(self, e, ctx):
        raise NotImplementedError('should not be called')

    def _visit_var_assign(self, stmt, ctx: _DefSet):
        return ctx | { stmt.var }

    def _visit_tuple_assign(self, stmt, ctx: _DefSet):
        return ctx | stmt.binding.names()

    def _visit_if_stmt(self, stmt, ctx: _DefSet):
        ift_defs = self._visit(stmt.ift, ctx)
        if stmt.iff is None:
            return ctx | ift_defs
        else:
            iff_defs = self._visit(stmt.iff, ctx)
            return ctx | ift_defs | iff_defs

    def _visit_while_stmt(self, stmt, ctx: _DefSet):
        block_defs = self._visit(stmt.body, ctx)
        return ctx | block_defs

    def _visit_for_stmt(self, stmt, ctx: _DefSet):
        block_defs = self._visit(stmt.body, ctx)
        return ctx | block_defs

    def _visit_return(self, stmt, ctx: _DefSet):
        return set(ctx)

    def _visit_block(self, block, ctx: _DefSet):
        def_in = ctx
        def_out: _DefSet = set()
        for stmt in block.stmts:
            def_out = self._visit(stmt, def_out)
        block.attribs[DefinitionAnalysis.analysis_name] = (def_in, def_out)
        return def_out

    def _visit_function(self, func, ctx: _DefSet):
        ctx = set(ctx)
        for arg in func.args:
            ctx.add(arg.name)
        self._visit(func.body, ctx)

class DefinitionAnalysis:
    """
    Definition analyzer on the AST.

    Associates to each statement block the set of variables defined in it.
    This is not the same as the set of variables in scope.
    """

    analysis_name = 'def_vars'

    @staticmethod
    def analyze(func: Function):
        """Analyze the defined variables in a function."""
        if not isinstance(func, Function):
            raise TypeError(f'expected a Function, got {func}')
        DefinitionAnalysisInstance(func).analyze()
