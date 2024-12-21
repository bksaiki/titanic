"""Definition-use analysis on the FPy IR"""

from ..ir import *

class _DefineUseInstance(DefaultVisitor):
    """Per-IR instance of definition-use analysis"""
    func: Function
    uses: dict[str, set[Var | PhiNode]]
    done: bool

    def __init__(self, func: Function):
        self.func = func
        self.uses = {}
        self.done = False

    def analyze(self):
        if self.done:
            raise RuntimeError('analysis already performed')
        self._visit(self.func, {})
        self.done = True
        return self.uses

    def _visit_var(self, e: Var, ctx):
        if e.name not in self.uses:
            raise NotImplementedError(f'undefined variable {e.name}')
        self.uses[e.name].add(e)

    def _visit_var_assign(self, stmt: VarAssign, ctx):
        self._visit(stmt.expr, ctx)
        self.uses[stmt.var] = set()

    def _visit_tuple_assign(self, stmt: TupleAssign, ctx):
        self._visit(stmt.expr, ctx)
        for var in stmt.vars.names():
            self.uses[var] = set()

    def _visit_if1_stmt(self, stmt, ctx):
        self._visit(stmt.cond, ctx)
        self._visit(stmt.body, ctx)
        for phi in stmt.phis:
            self.uses[phi.name] = set()
            self.uses[phi.lhs].add(phi)
            self.uses[phi.rhs].add(phi)

    def _visit_if_stmt(self, stmt, ctx):
        self._visit(stmt.cond, ctx)
        self._visit(stmt.ift, ctx)
        self._visit(stmt.iff, ctx)
        for phi in stmt.phis:
            self.uses[phi.name] = set()
            self.uses[phi.lhs].add(phi)
            self.uses[phi.rhs].add(phi)

    def _visit_while_stmt(self, stmt, ctx):
        for phi in stmt.phis:
            self.uses[phi.name] = set()
            self.uses[phi.lhs].add(phi)
        self._visit(stmt.cond, ctx)
        self._visit(stmt.body, ctx)
        for phi in stmt.phis:
            self.uses[phi.rhs].add(phi)

    def _visit_for_stmt(self, stmt, ctx):
        self._visit(stmt.iterable, ctx)
        self.uses[stmt.var] = set()
        for phi in stmt.phis:
            self.uses[phi.name] = set()
            self.uses[phi.lhs].add(phi)
        self._visit(stmt.body, ctx)
        for phi in stmt.phis:
            self.uses[phi.rhs].add(phi)

    def _visit_function(self, func: Function, ctx):
        for arg in func.args:
            self.uses[arg.name] = set()
        self._visit(func.body, ctx)

class DefineUse:
    """
    Definition-use analyzer for the FPy IR.

    Computes the set of definitions and their uses. Associates to
    each statement, the incoming definitions and outgoing definitions.
    """

    analysis_name = 'define_use'

    @staticmethod
    def analyze(func: Function):
        _DefineUseInstance(func).analyze()
