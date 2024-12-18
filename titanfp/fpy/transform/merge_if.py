"""Transformation pass to rewrite if statements to if expressions."""

from ..fpyast import *
from ..visitor import DefaultTransformVisitor
from ..analysis import DefUse, LiveVars
from ..analysis.def_use import DefUseEnv


class MergeIf(DefaultTransformVisitor):
    """
    Transforms if statements into if expressions.

    This rewriting pass transforms any if statement:
    ```
    if <cond>:
        S1 ...
    else:
        S2 ...
    S3 ...
    ```
    to an equivalent block using if expressions:
    ```
    t = <cond>
    S1 ...
    S2 ...
    x_i = x_{i, S1} if t else x_{i, S2}
    ...
    S3 ...
    ```
    where {`x_i`} are the set of live variables at the start of `S3`
    that are defined or mutated within `S1` or `S2`. Additionally,
    any definition in `S1` and `S2` is renamed to some free variable to
    avoid namespace collisions. Likewise, `t` is some free variable.
    """

    def_use: DefUse
    """Definition-use analysis instance."""

    live_vars: LiveVars
    """Live variable analysis instance."""

    def __init__(self):
        super().__init__()
        self.def_use = DefUse()
        self.live_vars = LiveVars()

    def _visit_block(self, block, ctx: DefUseEnv):
        new_stmts: list[Stmt] = []
        for i, stmt in enumerate(block.stmts):
            match stmt:
                case IfStmt():
                    assert i + 1 < len(block.stmts), 'if statement must be followed by a statement'
                    # variables in scope at the start of this statement
                    in_scope = ctx.keys()
                    # variables in scope at the start of each branch
                    ift_scope = stmt.ift.attribs[self.def_use.name].keys()
                    iff_scope = stmt.iff.attribs[self.def_use.name].keys()
                    # free variables at the start of the next statement
                    fvs = block.stmts[i+1].attribs[self.live_vars.name]

                    print(in_scope, ift_scope, iff_scope, fvs)


                    raise NotImplementedError(stmt, fvs)
                case _:
                    new_stmts.append(self._visit(stmt, ctx))
        return Block(new_stmts)

    # override typing hint
    def _visit(self, e, ctx: DefUseEnv):
        return super()._visit(e, ctx)

    def visit(self, e: Function | Block):
        if not (isinstance(e, Function) or isinstance(e, Block)):
            raise TypeError(f'visit() argument 1 must be Function or Block, not {e}')
        # definition-use analysis is required
        if self.def_use.name not in e.attribs:
            raise RuntimeError('must run DefUse analysis to use this transformation')
        if self.live_vars.name not in e.attribs:
            raise RuntimeError('must run LiveVars analysis to use this transformation')
        print(e)
        ctx: DefUseEnv = e.attribs[self.def_use.name]
        return self._visit(e, ctx)
