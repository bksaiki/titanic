"""
Basic syntax check to eliminate malformed FPy programs
that the parser can't detect.

Rules enforced:

- `return` statement:
  - all functions must have exactly one return statement
  - return statement must be at the end of the function definiton

- `if` statement:
any variable must be defined along both branches when
used after the `if` statement

- any variables must be defined before it is used

TODO: duplicate identifiers
"""

from ..fpyast import *
from ..visitor import Analysis
from ..utils import FPySyntaxError

class _VarEnv:
    """Bound variables in a scope."""
    _vars: dict[str, bool]

    def __init__(self):
        self._vars = dict()

    def __contains__(self, item):
        return item in self._vars

    def __getitem__(self, key):
        return self._vars[key]

    def add(self, var: str):
        """Creates a new scope by copying this scopye and adding `var` to it."""
        copy = _VarEnv()
        copy._vars = self._vars.copy()
        copy._vars[var] = True
        return copy
    
    def merge(self, other):
        if not isinstance(other, _VarEnv):
            raise TypeError('merge(): other argument must be of type \'_VarEnv\'', other)
        copy = _VarEnv()
        for name in self._vars.keys() | other._vars.keys():
            copy._vars[name] = self._vars.get(name, False) and other._vars.get(name, False)
        return copy


class SyntaxCheck(Analysis):
    """Visitor implementing syntax checking."""

    _CtxType = tuple[bool, _VarEnv]
    """
    Type of the `ctx` argument to each visitor method.

    Specifically,
    ```
    is_top, env = ctx
    ```
    with
    ```
    is_top: bool
    env: dict[str, str]
    ```
    where
    - `is_top`: is `True` if the visitor is at the top block of the function;
    - `env`: map from name to variable info, representing all variables in scope
    """

    def __init__(self):
        super().__init__()

    def _visit_binding(self, binding: Binding, env: _VarEnv):
        match binding:
            case VarBinding():
                return env.add(binding.name)
            case TupleBinding():
                for elt in binding.bindings:
                    env = self._visit_binding(elt, env)
                return env
            case _:
                raise NotImplementedError('unreachable', binding)

    def _visit_decnum(self, e, ctx: _CtxType):
        _, env = ctx
        return env

    def _visit_integer(self, e, ctx: _CtxType):
        _, env = ctx
        return env

    def _visit_digits(self, e, ctx: _CtxType):
        _, env = ctx
        return env

    def _visit_variable(self, e, ctx: _CtxType):
        _, env = ctx
        if e.name not in env:
            raise FPySyntaxError(f'unbound variable \'{e.name}\'')
        if not env[e.name]:
            raise FPySyntaxError(f'variable not defined on along all paths \'{e.name}\'')
        return env

    def _visit_unknown(self, e, ctx: _CtxType):
        _, env = ctx
        for c in e.children:
            self._visit(c, ctx)
        return env

    def _visit_nary_expr(self, e, ctx: _CtxType):
        _, env = ctx
        for c in e.children:
            self._visit(c, ctx)
        return env

    def _visit_compare(self, e, ctx: _CtxType):
        _, env = ctx
        for c in e.children:
            self._visit(c, ctx)
        return env

    def _visit_array(self, e, ctx: _CtxType):
        _, env = ctx
        for c in e.children:
            self._visit(c, ctx)
        return env

    def _visit_if_expr(self, e, ctx: _CtxType):
        _, env = ctx
        self._visit(e.cond, ctx)
        self._visit(e.ift, ctx)
        self._visit(e.iff, ctx)
        return env

    def _visit_assign(self, stmt, ctx: _CtxType):
        _, env = ctx
        self._visit(stmt.val, ctx)
        return self._visit_binding(stmt.var, env)

    def _visit_tuple_assign(self, stmt, ctx: _CtxType):
        _, env = ctx
        self._visit(stmt.val, ctx)
        return self._visit_binding(stmt.binding, env)
    
    def _visit_if_stmt(self, stmt, ctx: _CtxType):
        self._visit(stmt.cond, ctx)
        ift_env = self._visit(stmt.ift, ctx)
        iff_env = self._visit(stmt.iff, ctx)
        return ift_env.merge(iff_env)

    def _visit_return(self, stmt, ctx: _CtxType):
        return self._visit(stmt.e, ctx)

    def _visit_block(self, block, ctx: _CtxType):
        is_top, env = ctx
        has_return = False
        for i, st in enumerate(block.stmts):
            match st:
                case Assign():
                    env = self._visit_assign(st, (False, env))
                case TupleAssign():
                    env = self._visit_tuple_assign(st, (False, env))
                case Return():
                    if not is_top:
                        raise FPySyntaxError(f'return statement can only be at the top level')
                    if i != len(block.stmts) - 1:
                        raise FPySyntaxError(f'return statement must be a the end of a statement')
                    env = self._visit(st, (False, env))
                    has_return = True
                case IfStmt():
                    env = self._visit(st, (False, env))
                case _:
                    raise NotImplementedError('unreachable', st)

        if is_top and not has_return:
            raise FPySyntaxError(f'must have a return statement at the top-level')
        return env
    
    def _visit_function(self, func, ctx: _CtxType):
        env = _VarEnv()
        for arg in func.args:
            env = env.add(arg.name)
        self._visit(func.body, (True, env))

    # override typing hint
    def _visit(self, e, ctx: _CtxType) -> _VarEnv:
        return super()._visit(e, ctx)

    def visit(self, func: Function):
        if not isinstance(func, Function):
            raise TypeError(f'visit() argument 1 must be Function, not {func}')
        self._visit(func, (False, _VarEnv()))

