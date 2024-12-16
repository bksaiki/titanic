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
"""

from enum import Enum

from ..fpyast import *
from ..visitor import Visitor


class FPySyntaxError(Exception):
    """Syntax error for FPy programs."""
    pass

class SyntaxCheck(Visitor):
    """Visitor implementing syntax checking."""

    _CtxType = tuple[bool, dict[str, str]]
    """
    Type of the `ctx` visitor argument.

    Specifically,
    ```
    is_top, scope = ctx
    ```
    with
    ```
    is_top: bool
    scope: dict[str, str]
    ```
    where
    - `is_top`: is `True` if the visitor is at the top block of the function;
    - `scope`: map from name to variable info, representing all variables in scope
    """

    def _visit_decnum(self, e, ctx: _CtxType):
        pass

    def _visit_integer(self, e, ctx: _CtxType):
        pass

    def _visit_digits(self, e, ctx: _CtxType):
        pass

    def _visit_variable(self, e, ctx: _CtxType):
        _, scope = ctx
        if e.name not in scope:
            raise FPySyntaxError(f'unbound variable \'{e.name}\'')

    def _visit_unknown(self, e, ctx: _CtxType):
        for c in e.children:
            self._visit(c, ctx)

    def _visit_nary_expr(self, e, ctx: _CtxType):
        for c in e.children:
            self._visit(c, ctx)

    def _visit_compare(self, e, ctx: _CtxType):
        for c in e.children:
            self._visit(c, ctx)

    def _visit_array(self, e, ctx: _CtxType):
        for c in e.children:
            self._visit(c, ctx)

    def _visit_if_expr(self, e, ctx: _CtxType):
        self._visit(e.cond, ctx)
        self._visit(e.ift, ctx)
        self._visit(e.iff, ctx)

    def _visit_assign(self, stmt, ctx: _CtxType):
        _, scope = ctx
        self._visit(stmt.val, ctx)
        return {**scope, stmt.var.name: stmt.var.name }

    def _visit_tuple_assign(self, stmt, ctx: _CtxType):
        raise NotImplementedError
    
    def _visit_if_stmt(self, stmt, ctx: _CtxType):
        self._visit(stmt.cond, ctx)
        self._visit(stmt.ift, ctx)
        self._visit(stmt.iff, ctx)

    def _visit_return(self, stmt, ctx: _CtxType):
        self._visit(stmt.e, ctx)

    def _visit_block(self, stmt, ctx: _CtxType):
        is_top, scope = ctx
        has_return = False
        for i, st in enumerate(stmt.stmts):
            match st:
                case Assign():
                    scope = self._visit_assign(st, (False, scope))
                case TupleAssign():
                    scope = self._visit_tuple_assign(st, (False, scope))
                case Return():
                    if not is_top:
                        raise FPySyntaxError(f'return statement can only be at the top level')
                    if i != len(stmt.stmts) - 1:
                        raise FPySyntaxError(f'return statement must be a the end of a statement')
                    self._visit(st, (False, scope))
                    has_return = True
                case IfStmt():
                    self._visit(st, (False, scope))
                case Block():
                    raise NotImplementedError('cannot have an internal block', stmt)
                case _:
                    raise NotImplementedError('unreachable', st)

        if is_top and not has_return:
            raise FPySyntaxError(f'must have a return statement at the top-level')
    
    def _visit_function(self, func, ctx: _CtxType):
        self._visit(func.body, (True, dict()))

    # To override typing hint
    def _visit(self, e, ctx: _CtxType) -> None:
        return super()._visit(e, ctx)

    def visit(self, func: Function):
        if not isinstance(func, Function):
            raise TypeError(f'visit() argument 1 must be Function, not {func}')
        self._visit(func, (False, dict()))

