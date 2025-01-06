"""Syntax checking for the FPy AST."""

from typing import Optional, Self

from .fpyast import *
from .visitor import AstVisitor

from ..utils import FPySyntaxError

class _Env:
    """Bound variables in the current scope."""
    env: dict[str, bool]

    def __init__(self, env: Optional[dict[str, bool]] = None):
        if env is None:
            self.env = {}
        else:
            self.env = env.copy()

    def __contains__(self, key):
        return key in self.env

    def __getitem__(self, key):
        return self.env[key]

    def extend(self, var: str):
        copy = _Env(self.env)
        copy.env[var] = True
        return copy

    def merge(self, other: Self):
        copy = _Env()
        for key in self.env.keys() | other.env.keys():
            copy.env[key] = self.env.get(key, False) and other.env.get(key, False)
        return copy

_Ctx = tuple[_Env, bool]
"""
1st element: environment
2nd element: whether the current block is at the top-level.
"""

class SyntaxCheckInstance(AstVisitor):
    """Single-use instance of syntax checking"""
    func: Function

    def __init__(self, func: Function):
        self.func = func

    def analyze(self):
        self._visit(self.func, (_Env(), False))

    def _visit_var(self, e, ctx: _Ctx):
        env, _ = ctx
        if e.name not in env:
            raise FPySyntaxError(f'unbound variable `{e.name}`')
        if not env[e.name]:
            raise FPySyntaxError(f'variable `{e.name}` not defined along all paths')
        return env

    def _visit_decnum(self, e, ctx: _Ctx):
        env, _ = ctx
        return env

    def _visit_integer(self, e, ctx: _Ctx):
        env, _ = ctx
        return env

    def _visit_unaryop(self, e, ctx: _Ctx):
        env, _ = ctx
        self._visit(e.arg, ctx)
        return env

    def _visit_binaryop(self, e, ctx: _Ctx):
        env, _ = ctx
        self._visit(e.left, ctx)
        self._visit(e.right, ctx)
        return env

    def _visit_ternaryop(self, e, ctx: _Ctx):
        env, _ = ctx
        self._visit(e.arg0, ctx)
        self._visit(e.arg1, ctx)
        self._visit(e.arg2, ctx)
        return env

    def _visit_naryop(self, e, ctx: _Ctx):
        env, _ = ctx
        for c in e.args:
            self._visit(c, ctx)
        return env

    def _visit_compare(self, e, ctx: _Ctx):
        env, _ = ctx
        for c in e.args:
            self._visit(c, ctx)
        return env

    def _visit_call(self, e, ctx: _Ctx):
        env, _ = ctx
        for c in e.args:
            self._visit(c, ctx)
        return env

    def _visit_tuple_expr(self, e, ctx: _Ctx):
        env, _ = ctx
        for c in e.args:
            self._visit(c, ctx)
        return env

    def _visit_comp_expr(self, e, ctx: _Ctx):
        env, _ = ctx
        for iterable in e.iterables:
            self._visit(iterable, ctx)
        for var in e.vars:
            env = env.extend(var)
        self._visit(e.elt, (env, False))
        return env

    def _visit_if_expr(self, e, ctx: _Ctx):
        env, _ = ctx
        self._visit(e.cond, ctx)
        self._visit(e.ift, ctx)
        self._visit(e.iff, ctx)
        return env

    def _visit_var_assign(self, stmt, ctx: _Ctx):
        env, _ = ctx
        self._visit(stmt.expr, ctx)
        return env.extend(stmt.var)
    
    def _visit_tuple_binding(self, binding: TupleBinding, ctx: _Ctx):
        env, _ = ctx
        for elt in binding.elts:
            match elt:
                case str():
                    env = env.extend(elt)
                case TupleBinding():
                    env = self._visit_tuple_binding(elt, ctx)
                case _:
                    raise NotImplementedError('unreachable', elt)
        return env

    def _visit_tuple_assign(self, stmt, ctx: _Ctx):
        self._visit(stmt.expr, ctx)
        return self._visit_tuple_binding(stmt.binding, ctx)

    def _visit_if_stmt(self, stmt, ctx: _Ctx):
        self._visit(stmt.cond, ctx)
        ift_env = self._visit(stmt.ift, ctx)
        if stmt.iff is None:
            # 1-armed if statement
            env, _ = ctx
            return ift_env.merge(env)
        else:
            iff_env = self._visit(stmt.iff, ctx)
            return ift_env.merge(iff_env)

    def _visit_while_stmt(self, stmt, ctx: _Ctx):
        env, _ = ctx
        body_env = self._visit(stmt.body, ctx)
        env = env.merge(body_env)
        self._visit(stmt.cond, (env, False))
        return env

    def _visit_for_stmt(self, stmt, ctx: _Ctx):
        env, _ = ctx
        self._visit(stmt.iterable, ctx)
        env = env.extend(stmt.var)
        body_env = self._visit(stmt.body, (env, False))
        return env.merge(body_env)

    def _visit_return(self, stmt, ctx: _Ctx):
        return self._visit(stmt.expr, ctx)

    def _visit_block(self, block, ctx: _Ctx):
        env, is_top = ctx
        has_return = False
        for i, stmt in enumerate(block.stmts):
            match stmt:
                case VarAssign():
                    env = self._visit(stmt, (env, False))
                case TupleAssign():
                    env = self._visit(stmt, (env, False))
                case IfStmt():
                    env = self._visit(stmt, (env, False))
                case WhileStmt():
                    env = self._visit(stmt, (env, False))
                case ForStmt():
                    env = self._visit(stmt, (env, False))
                case Return():
                    if not is_top:
                        raise FPySyntaxError(f'return statement must be at the top-level')
                    if i != len(block.stmts) - 1:
                        raise FPySyntaxError(f'return statement must be at the end of the function definition')
                    env = self._visit(stmt, (env, False))
                    has_return = True
                case _:
                    raise NotImplementedError('unreachable', stmt)

        if is_top and not has_return:
            raise FPySyntaxError(f'must have a return statement at the top-level')
        return env

    def _visit_function(self, func, ctx: _Ctx):
        env, _ = ctx
        for arg in func.args:
            env = env.extend(arg.name)
        return self._visit(func.body, (env, True))

    # override for typing hint
    def _visit(self, e, ctx: _Ctx) -> _Env:
        return super()._visit(e, ctx)


class SyntaxCheck:
    """
    Syntax checker for the FPy AST.

    Basic syntax check to eliminate malformed FPy programs that
    the parser can't detect.

    Rules enforced:

    - any variables must be defined before it is used;

    - `return` statement:
      - all functions must have exactly one return statement,
      - return statement must be at the end of the function definiton;

    - `if` statement:
      - any variable must be defined along both branches when
        used after the `if` statement
    """

    @staticmethod
    def analyze(func: Function):
        if not isinstance(func, Function):
            raise TypeError(f'expected a Function, got {func}')
        SyntaxCheckInstance(func).analyze()
