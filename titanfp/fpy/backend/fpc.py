"""Compilation from FPy IR to FPCore"""

from typing import Optional

from ..passes import *
from ..ir import *
from ..utils import Gensym

from ...fpbench import fpcast as fpc

_op_table = {
    '+': fpc.Add,
    '-': fpc.Sub,
    '*': fpc.Mul,
    '/': fpc.Div,
    'fabs': fpc.Fabs,
    'sqrt': fpc.Sqrt,
    'fma': fpc.Fma,
    'neg': fpc.Neg,
    'copysign': fpc.Copysign,
    'fdim': fpc.Fdim,
    'fmax': fpc.Fmax,
    'fmin': fpc.Fmin,
    'fmod': fpc.Fmod,
    'remainder': fpc.Remainder,
    'hypot': fpc.Hypot,
    'cbrt': fpc.Cbrt,
    'ceil': fpc.Ceil,
    'floor': fpc.Floor,
    'nearbyint': fpc.Nearbyint,
    'round': fpc.Round,
    'trunc': fpc.Trunc,
    'acos': fpc.Acos,
    'asin': fpc.Asin,
    'atan': fpc.Atan,
    'atan2': fpc.Atan2,
    'cos': fpc.Cos,
    'sin': fpc.Sin,
    'tan': fpc.Tan,
    'acosh': fpc.Acosh,
    'asinh': fpc.Asinh,
    'atanh': fpc.Atanh,
    'cosh': fpc.Cosh,
    'sinh': fpc.Sinh,
    'tanh': fpc.Tanh,
    'exp': fpc.Exp,
    'exp2': fpc.Exp2,
    'expm1': fpc.Expm1,
    'log': fpc.Log,
    'log10': fpc.Log10,
    'log1p': fpc.Log1p,
    'log2': fpc.Log2,
    'pow': fpc.Pow,
    'erf': fpc.Erf,
    'erfc': fpc.Erfc,
    'lgamma': fpc.Lgamma,
    'tgamma': fpc.Tgamma,
    'isfinite': fpc.Isfinite,
    'isinf': fpc.Isinf,
    'isnan': fpc.Isnan,
    'isnormal': fpc.Isnormal,
    'signbit': fpc.Signbit,
    'not': fpc.Not,
    'or': fpc.Or,
    'and': fpc.And,
}

class FPCoreCompileError(Exception):
    """Any FPCore compilation error"""
    pass

class FPCoreCompileInstance(ReduceVisitor):
    """Compilation instance from FPy to FPCore"""
    func: Function
    gensym: Gensym

    def __init__(self, func: Function):
        uses = DefineUse().analyze(func)
        self.func = func
        self.gensym = Gensym(*uses.keys())

    def compile(self) -> fpc.FPCore:
        f = self._visit(self.func, None)
        assert isinstance(f, fpc.FPCore), 'unexpected result type'
        return f
    
    def _compile_arg(self, arg: Argument):
        match arg.ty:
            case RealType():
                return arg.name, None, None
            case AnyType():
                return arg.name, None, None
            case _:
                raise FPCoreCompileError('unsupported argument type', arg)

    def _compile_tuple_binding(self, tuple_id: str, binding: TupleBinding, pos: list[int]):
        tuple_binds: list[tuple[str, fpc.Expr]] = []
        for i, elt in enumerate(binding):
            match elt:
                case str():
                    idxs = [fpc.Integer(idx) for idx in [i, *pos]]
                    tuple_binds.append((elt, fpc.Ref(fpc.Var(tuple_id), *idxs)))
                case TupleBinding():
                    tuple_binds += self._compile_tuple_binding(tuple_id, elt, [i, *pos])
                case _:
                    raise FPCoreCompileError('unexpected tensor element', elt)
        return tuple_binds

    def _compile_compareop(self, op: CompareOp):
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
                raise NotImplementedError('unreachable', op)

    def _visit_var(self, e, ctx) -> fpc.Expr:
        return fpc.Var(e.name)

    def _visit_decnum(self, e, ctx) -> fpc.Expr:
        return fpc.Decnum(e.val)

    def _visit_integer(self, e, ctx) -> fpc.Expr:
        return fpc.Integer(e.val)

    def _visit_digits(self, e, ctx) -> fpc.Expr:
        return fpc.Digits(e.m, e.e, e.b)

    def _visit_unknown(self, e, ctx) -> fpc.Expr:
        args = [self._visit(c, ctx) for c in e.children]
        return fpc.UnknownOperator(e.name, *args)

    def _visit_nary_expr(self, e, ctx) -> fpc.Expr:
        if e.name == Range.name:
            # expand range expression
            tuple_id = 'i'
            size = self._visit(e.children[0], ctx)
            return fpc.Tensor([(tuple_id, size)], fpc.Var(tuple_id))
        else:
            cls = _op_table.get(e.name)
            if cls is None:
                raise NotImplementedError('no FPCore operator for', e.name)
            return cls(*[self._visit(c, ctx) for c in e.children])

    def _visit_compare(self, e, ctx) -> fpc.Expr:
        assert e.ops != [], 'should not be empty'
        match e.ops:
            case [op]:
                # 2-argument case: just compile
                cls = self._compile_compareop(op)
                arg0 = self._visit(e.children[0], ctx)
                arg1 = self._visit(e.children[1], ctx)
                return cls(arg0, arg1)
            case [op, *ops]:
                # N-argument case:
                # TODO: want to evaluate each argument only once;
                #       may need to let-bind in case any argument is
                #       used multiple times
                args = [self._visit(arg, ctx) for arg in e.children]
                curr_group = (op, [args[0], args[1]])
                groups: list[tuple[CompareOp, list[fpc.Expr]]] = [curr_group]
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
                    cls = self._compile_compareop(op)
                    return cls(*args)
                else:
                    args = [self._compile_compareop(op)(*args) for op, args in groups]
                    return fpc.And(*args)
            case _:
                raise NotImplementedError('unreachable', e.ops)

    def _visit_tuple_expr(self, e, ctx) -> fpc.Expr:
        return fpc.Array(*[self._visit(c, ctx) for c in e.children])

    def _visit_ref_expr(self, e, ctx) -> fpc.Expr:
        array = self._visit(e.array, ctx)
        indices = [self._visit(c, ctx) for c in e.indices]
        return fpc.Ref(array, *indices)

    def _visit_comp_expr(self, e, ctx):
        tuple_id = self.gensym.fresh('t')
        iter_id = self.gensym.fresh('i')
        iterable = self._visit(e.iterable, ctx)
        elt = self._visit(e.elt, ctx)

        let_bindings = [(tuple_id, iterable)]
        tensor_dims = [(iter_id, fpc.Size(tuple_id))]
        ref_bindings = [(e.var, fpc.Ref(fpc.Var(tuple_id), fpc.Var(iter_id)))]
        return fpc.Let(let_bindings, fpc.Tensor(tensor_dims, fpc.Let(ref_bindings, elt)))

    def _visit_if_expr(self, e, ctx) -> fpc.Expr:
        return fpc.If(
            self._visit(e.cond, ctx),
            self._visit(e.ift, ctx),
            self._visit(e.iff, ctx)
        )

    def _visit_var_assign(self, stmt: VarAssign, ctx: fpc.Expr):
        bindings = [(stmt.var, self._visit(stmt.expr, None))]
        return fpc.Let(bindings, ctx)

    def _visit_tuple_assign(self, stmt: TupleAssign, ctx: fpc.Expr):
        tuple_id = self.gensym.fresh('t')
        tuple_bind = (tuple_id, self._visit(stmt.expr, None))
        destruct_bindings = self._compile_tuple_binding(tuple_id, stmt.binding, [])
        return fpc.Let([tuple_bind] + destruct_bindings, ctx)

    def _visit_if1_stmt(self, stmt, ctx):
        raise FPCoreCompileError(f'cannot compile to FPCore: {type(stmt).__name__}')

    def _visit_if_stmt(self, stmt, ctx):
        raise FPCoreCompileError(f'cannot compile to FPCore: {type(stmt).__name__}')

    def _visit_while_stmt(self, stmt, ctx: fpc.Expr):
        if len(stmt.phis) != 1:
            raise FPCoreCompileError('while loops must have exactly one phi node')
        phi = stmt.phis[0]
        name, init, update = phi.name, phi.lhs, phi.rhs
        cond = self._visit(stmt.cond, None)
        body = self._visit(stmt.body, fpc.Var(update))
        return fpc.While(cond, [(name, fpc.Var(init), body)], ctx)

    def _visit_for_stmt(self, stmt, ctx):
        if len(stmt.phis) != 1:
            raise FPCoreCompileError('for loops must have exactly one phi node')
        # phi nodes
        phi = stmt.phis[0]
        name, init, update = phi.name, phi.lhs, phi.rhs
        # fresh variable for the iterable value
        tuple_id = self.gensym.fresh('t')
        iterable = self._visit(stmt.iterable, None)
        body = self._visit(stmt.body, fpc.Var(update))
        # index variables and state merging
        dim_binding = (stmt.var, fpc.Size(tuple_id))
        while_binding = (name, fpc.Var(init), body)
        return fpc.Let([(tuple_id, iterable)], fpc.For([dim_binding], [while_binding], ctx))

    def _visit_return(self, stmt, ctx) -> fpc.Expr:
        return self._visit(stmt.expr, ctx)

    def _visit_block(self, block, ctx: Optional[fpc.Expr]):
        if ctx is None:
            # entering from the top-level
            ret_stmt = block.stmts[-1]
            if not isinstance(ret_stmt, Return):
                raise FPCoreCompileError('blocks must have a return statement at the end')
            e = self._visit(ret_stmt, ctx)
            stmts = block.stmts[:-1]
        else:
            # entering from a nested block
            e = ctx
            stmts = block.stmts

        for stmt in reversed(stmts):
            match stmt:
                case VarAssign() | TupleAssign() | WhileStmt() | ForStmt():
                    e = self._visit(stmt, e)
                case Return():
                    raise FPCoreCompileError('return statements must be at the end of blocks')
                case _:
                    raise FPCoreCompileError(f'cannot compile to FPCore: {type(stmt).__name__}')
        return e

    def _visit_function(self, func, ctx):
        args = [self._compile_arg(arg) for arg in func.args]
        # TODO: parse data

        # compile body
        e = self._visit(func.body, ctx)

        return fpc.FPCore(
            ident=func.name,
            inputs=args,
            e=e,
        )

    # override for typing hint
    def _visit(self, e, ctx) -> fpc.Expr:
        return super()._visit(e, ctx)

class FPCoreCompiler:
    """Compiler from FPy IR to FPCore"""

    def compile(self, func: Function) -> fpc.FPCore:
        func = ForBundling.apply(func)
        func = WhileBundling.apply(func)
        func = SimplifyIf.apply(func)
        return FPCoreCompileInstance(func).compile()
