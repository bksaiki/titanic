"""Pass to ensure correctness of the FPy IR."""

from ..ir import *

_CtxType = set[str]

class InvalidIRError(Exception):
    pass

# TODO: type check
class _VerifyPassInstance(DefaultVisitor):
    """Single instance of the `VerifyPass`."""
    func: Function
    types: dict[str, IRType]

    def __init__(self, func: Function):
        self.func = func
        self.types = {}

    def check(self):
        self.types = {}
        self._visit(self.func, set())

    def _visit_var(self, e, ctx: _CtxType):
        if e.name not in ctx:
            raise InvalidIRError(f'undefined variable {e.name}')

    def _visit_comp_expr(self, e, ctx: _CtxType):
        for iterable in e.iterables:
            self._visit(iterable, ctx)
        for var in e.vars:
            if var in self.types:
                raise InvalidIRError(f'reassignment of variable {var}')
            self.types[var] = AnyType()
            ctx.add(var)
        self._visit(e.elt, ctx)

    def _visit_var_assign(self, stmt, ctx: _CtxType):
        self._visit(stmt.expr, ctx)
        if stmt.var in self.types:
            raise InvalidIRError(f'reassignment of variable {stmt.var}')
        self.types[stmt.var] = AnyType()
        ctx.add(stmt.var)
        return ctx

    def _visit_tuple_assign(self, stmt, ctx: _CtxType):
        self._visit(stmt.expr, ctx)
        for var in stmt.binding.names():
            if var in self.types:
                raise InvalidIRError(f'reassignment of variable {var}')
            self.types[var] = AnyType()
            ctx.add(var)
        return ctx

    def _visit_if1_stmt(self, stmt, ctx: _CtxType):
        self._visit(stmt.cond, ctx)
        body_ctx = self._visit_block(stmt.body, ctx.copy())
        # check validty of phi nodes and update context
        for phi in stmt.phis:
            name, orig, new = phi.name, phi.lhs, phi.rhs
            if name in self.types:
                raise InvalidIRError(f'reassignment of variable {name}')
            if orig not in ctx:
                raise InvalidIRError(f'undefined variable in LHS of phi {name} = ({orig}, {new})')
            if new not in body_ctx:
                raise InvalidIRError(f'undefined variable in RHS of phi {name} = ({orig}, {new})')
            if new == name:
                raise InvalidIRError(f'phi variable assigned to itself {name} = ({orig}, {new})')
            self.types[name] = AnyType()
            ctx.add(name)
            ctx -= { orig, new }
        return ctx

    def _visit_if_stmt(self, stmt, ctx: _CtxType):
        self._visit(stmt.cond, ctx)
        ift_ctx = self._visit_block(stmt.ift, ctx.copy())
        iff_ctx = self._visit_block(stmt.iff, ctx.copy())
        # check validty of phi nodes and update context
        for phi in stmt.phis:
            name, ift_name, iff_name = phi.name, phi.lhs, phi.rhs
            if name in self.types:
                raise InvalidIRError(f'reassignment of variable {name}')
            if ift_name not in ift_ctx:
                raise InvalidIRError(f'undefined variable in LHS of phi {name} = ({ift_name}, {iff_name})')
            if iff_name not in iff_ctx:
                raise InvalidIRError(f'undefined variable in RHS of phi {name} = ({ift_name}, {iff_name})')
            if ift_name == iff_name:
                raise InvalidIRError(f'phi variable is unnecessary {name} = ({ift_name}, {iff_name})')
            self.types[name] = AnyType()
            ctx.add(name)
            ctx -= { ift_name, iff_name }
        return ctx

    def _visit_while_stmt(self, stmt, ctx: _CtxType):
        # check (partial) validity of phi variables and update context
        for phi in stmt.phis:
            name, orig = phi.name, phi.lhs
            if name in self.types:
                raise InvalidIRError(f'reassignment of variable {name}')
            if orig not in ctx:
                raise InvalidIRError(f'undefined variable in LHS of phi {name} = ({orig}, _)')
            self.types[name] = AnyType()
            ctx.add(name)
            ctx -= { orig }
        # check condition and body
        self._visit(stmt.cond, ctx)
        body_ctx = self._visit_block(stmt.body, ctx.copy())
        # check (partial) validity of phi variables
        for phi in stmt.phis:
            name, orig, new = phi.name, phi.lhs, phi.rhs
            if new not in body_ctx:
                raise InvalidIRError(f'undefined variable in RHS of phi {name} = (_, {new})')
            if new == name:
                raise InvalidIRError(f'phi variable assigned to itself {name} = ({orig}, {new})')
            ctx -= { new }
        return ctx

    def _visit_for_stmt(self, stmt, ctx: _CtxType):
        # check iterable expression
        self._visit(stmt.iterable, ctx)
        # bind the loop variable
        if stmt.var in self.types:
            raise InvalidIRError(f'reassignment of variable {stmt.var}')
        self.types[stmt.var] = AnyType()
        ctx.add(stmt.var)
        # check (partial) validity of phi variables and update context
        for phi in stmt.phis:
            name, orig = phi.name, phi.lhs
            if name in self.types:
                raise InvalidIRError(f'reassignment of variable {name}')
            if orig not in ctx:
                raise InvalidIRError(f'undefined variable in LHS of phi {name} = ({orig}, _)')
            self.types[name] = AnyType()
            ctx.add(name)
            ctx -= { orig }
        # check body
        body_ctx = self._visit_block(stmt.body, ctx.copy())
        # check (partial) validity of phi variables
        for phi in stmt.phis:
            name, orig, new = phi.name, phi.lhs, phi.rhs
            if new not in body_ctx:
                raise InvalidIRError(f'undefined variable in RHS of phi {name} = (_, {new})')
            if new == name:
                raise InvalidIRError(f'phi variable assigned to itself {name} = ({orig}, {new})')
            ctx -= { new }
        return ctx

    def _visit_block(self, block, ctx: _CtxType):
        for stmt in block.stmts:
            if isinstance(stmt, Return):
                self._visit_return(stmt, ctx)
                ctx = set()
            else:
                ctx = self._visit(stmt, ctx.copy())
        return ctx

    def _visit_function(self, func, ctx: _CtxType):
        for arg in func.args:
            self.types[arg.name] = AnyType()
            ctx.add(arg.name)
        self._visit(func.body, ctx)

    # override typing hint
    def _visit(self, e, ctx: _CtxType) -> Any:
        return super()._visit(e, ctx)


class VerifyIR:
    """
    Checks that an FPy IR instance is syntactically valid,
    well-typed, and in static single assignment (SSA) form.
    """

    @staticmethod
    def check(func: Function):
        _VerifyPassInstance(func).check()
