"""Compilation from FPy to FPCore"""

from ..fpbench import fpcast as fpc
from .ops import op_info

from .fpyast import *
from .visitor import ReduceVisitor
from .transform import *

class FPCoreCompiler(ReduceVisitor):
    """
    Compiler from FPy to FPCore programs.
    """
    
    def _visit_argument(self, arg: Argument):
        match arg.ty:
            case RealType():
                return arg.name, None, None
            case _:
                raise NotImplementedError(arg)

    def _make_tuple_binding(self, tuple_id: str, binding: TupleBinding, pos: list[int]):
        tuple_binds: list[tuple[str, fpc.Expr]] = []
        for i, bind in enumerate(binding.bindings):
            match bind:
                case TupleBinding():
                    tuple_binds += self._make_tuple_binding(tuple_id, bind, [i, *pos])
                case VarBinding(name=name):
                    idxs = [fpc.Integer(idx) for idx in [i, *pos]]
                    tuple_binds.append((name, fpc.Ref(fpc.Var(tuple_id), *idxs)))
                case _:
                    raise NotImplementedError('unexpected binding', bind)
        return tuple_binds
    
    def _visit_compareop(self, op: CompareOp):
        match op:
            case CompareOp.LT:
                return fpc.LT
            case CompareOp.LE:
                return fpc.LEQ
            case CompareOp.GE:
                return fpc.GEQ
            case CompareOp.GT:
                return fpc.GT
            case CompareOp.EQ:
                return fpc.EQ
            case CompareOp.NE:
                return fpc.NEQ
            case _:
                raise NotImplementedError(op)

    #######################################################
    # Expressions

    def _visit_decnum(self, e, ctx):
        return fpc.Decnum(e.val)
    
    def _visit_integer(self, e, ctx):
        return fpc.Integer(e.val)
    
    def _visit_digits(self, e, ctx):
        return fpc.Digits(e.m, e.e, e.b)
    
    def _visit_variable(self, e, ctx):
        return fpc.Var(e.name)
    
    def _visit_array(self, e, ctx):
        args = [self._visit(c, ctx) for c in e.children]
        return fpc.Array(*args)
    
    def _visit_unknown(self, e, ctx):
        args = [self._visit(c, ctx) for c in e.children]
        return fpc.UnknownOperator(e.name, args)
    
    def _visit_nary_expr(self, e, ctx):
        info = op_info(e.name)
        if info is None:
            raise NotImplementedError('no compilation method for', e)
        return info.fpc(*[self._visit(c, ctx) for c in e.children])
    
    def _visit_compare(self, e, ctx):
        assert e.ops != [], 'should not be empty'
        match e.ops:
            case [op]:
                # 2-argument case: just compile
                cls = self._visit_compareop(op)
                arg0 = self._visit(e.children[0], ctx)
                arg1 = self._visit(e.children[1], ctx)
                return cls(arg0, arg1)
            case [op, *ops]:
                # N-argument case:
                # TODO: want to evaluate each argument only once;
                #       may need to let-bind in case any argument is
                #       used multiple times
                args = [self._visit(arg, ctx) for arg in e.children]

                # gensym = Gensym()
                # for arg in args:
                #     if isinstance(arg, fpc.Var):
                #         gensym.reserve(arg.name)

                curr_group = (op, [args[0], args[1]])
                groups: list[tuple[CompareOp, list[fpc.Expr]]] = [curr_group]
                # let_binds: dict[str, fpc.Expr] = {}

                for op, lhs, rhs in zip(ops, args[1:], args[2:]):
                    if op == curr_group[0] or isinstance(lhs, fpc.ValueExpr):
                        # same op => append
                        # different op (terminal) => append
                        curr_group[1].append(lhs)
                    else:
                        # different op (non-terminal) => new group
                        new_group = (op, [lhs, rhs])
                        groups.append(new_group)
                        curr_group = new_group
                    
                if len(groups) == 1:
                    op, args = groups[0]
                    cls = self._visit_compareop(op)
                    return cls(*args)
                else:
                    args = [self._visit_compareop(op)(*args) for op, args in groups]
                    return fpc.And(*args)
    
    def _visit_if_expr(self, e, ctx):
        return fpc.If(
            self._visit(e.cond, ctx),
            self._visit(e.ift, ctx),
            self._visit(e.iff, ctx)
        )

    # overload to get typing hint
    def _visit_expr(self, e, ctx) -> fpc.Expr:
        return super()._visit_expr(e, ctx)

    #######################################################
    # Statements

    def _visit_assign(self, stmt, ctx):
        bindings = [(stmt.var.name, self._visit(stmt.val, ctx))]
        return (fpc.Let, bindings)
    
    def _visit_tuple_assign(self, stmt, ctx):
        tuple_id = 't0' # TODO: needs to be a unique identifier
        tuple_bind = (tuple_id, self._visit(stmt.val, ctx))
        destruct_bindings = self._make_tuple_binding(tuple_id, stmt.binding, [])
        return (fpc.Let, [tuple_bind] + destruct_bindings)
    
    def _visit_return(self, stmt, ctx):
        return self._visit(stmt.e, ctx)
    
    def _visit_if_stmt(self, stmt, ctx):
        raise NotImplementedError
    
    def _visit_phi(self, stmt, ctx):
        raise NotImplementedError
    
    def _visit_block(self, block, ctx):
        def _build(stmts: list[Stmt]) -> fpc.Expr:
            assert stmts != [], 'block is unexpectedly empty'
            match stmts[0]:
                case Assign():
                    cls, bindings = self._visit_assign(stmts[0], ctx)
                    return cls(bindings, _build(stmts[1:]))
                case TupleAssign():
                    cls, bindings = self._visit_tuple_assign(stmts[0], ctx)
                    return cls(bindings, _build(stmts[1:]))
                case Return():
                    assert stmts[1:] == [], 'return statements must be at the end of blocks'
                    return self._visit_return(stmts[0], ctx)
                case _:
                    raise NotImplementedError('unreachable', stmts[0])
        return _build(block.stmts)

    #######################################################
    # Functions

    def _visit_function(self, func, ctx):
        args = [self._visit_argument(arg) for arg in func.args]
        # TODO: parse data
        props = func.ctx.props

        # compile body
        e = self._visit(func.body, ctx)

        return fpc.FPCore(
            inputs=args,
            e=e,
            props=props,
            ident=func.ident,
            name=func.name,
            pre=func.pre
        )
    
    #######################################################
    # Entry-point

    def visit(self, f: Function):
        if not isinstance(f, Function):
            raise TypeError(f'expected Function: {f}')

        # Normalizing transformations
        f, replace_dict = SSA().visit(f)
        f = MergeIf().visit(f)

        return self._visit_function(f, None)


def fpy_to_fpcore(func: Function):
    compiler = FPCoreCompiler()
    return compiler.visit(func)
