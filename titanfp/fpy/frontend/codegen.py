"""
This module does intermediate code generation, compiling
the abstract syntax tree (AST) to the intermediate representation (IR).
"""

from .definition import DefinitionAnalysis
from .fpyast import *
from .live_vars import LiveVarAnalysis
from .visitor import AstVisitor

from .. import ir
from ..utils import Gensym

_CtxType = dict[str, str]

class _IRCodegenInstance(AstVisitor):
    """Instance of lowering an AST to an IR."""
    func: Function
    gensym: Gensym

    def __init__(self, func: Function):
        self.func = func
        self.gensym = Gensym()

    def lower(self) -> ir.Function:
        return self._visit(self.func, {})

    def _visit_var(self, e, ctx: _CtxType):
        return ir.Var(ctx[e.name])

    def _visit_decnum(self, e, ctx: _CtxType):
        return ir.Decnum(e.val)

    def _visit_integer(self, e, ctx: _CtxType):
        return ir.Integer(e.val)

    def _visit_unaryop(self, e, ctx: _CtxType):
        match e.op:
            case UnaryOpKind.NEG:
                arg = self._visit(e.arg, ctx)
                return ir.Neg(arg)
            case UnaryOpKind.NOT:
                arg = self._visit(e.arg, ctx)
                return ir.Not(arg)
            case UnaryOpKind.FABS:
                arg = self._visit(e.arg, ctx)
                return ir.Fabs(arg)
            case UnaryOpKind.SQRT:
                arg = self._visit(e.arg, ctx)
                return ir.Sqrt(arg)
            case UnaryOpKind.CBRT:
                arg = self._visit(e.arg, ctx)
                return ir.Cbrt(arg)
            case UnaryOpKind.CEIL:
                arg = self._visit(e.arg, ctx)
                return ir.Ceil(arg)
            case UnaryOpKind.FLOOR:
                arg = self._visit(e.arg, ctx)
                return ir.Floor(arg)
            case UnaryOpKind.NEARBYINT:
                arg = self._visit(e.arg, ctx)
                return ir.Nearbyint(arg)
            case UnaryOpKind.ROUND:
                arg = self._visit(e.arg, ctx)
                return ir.Round(arg)
            case UnaryOpKind.TRUNC:
                arg = self._visit(e.arg, ctx)
                return ir.Trunc(arg)
            case UnaryOpKind.ACOS:
                arg = self._visit(e.arg, ctx)
                return ir.Acos(arg)
            case UnaryOpKind.ASIN:
                arg = self._visit(e.arg, ctx)
                return ir.Asin(arg)
            case UnaryOpKind.ATAN:
                arg = self._visit(e.arg, ctx)
                return ir.Atan(arg)
            case UnaryOpKind.COS:
                arg = self._visit(e.arg, ctx)
                return ir.Cos(arg)
            case UnaryOpKind.SIN:
                arg = self._visit(e.arg, ctx)
                return ir.Sin(arg)
            case UnaryOpKind.TAN:
                arg = self._visit(e.arg, ctx)
                return ir.Tan(arg)
            case UnaryOpKind.ACOSH:
                arg = self._visit(e.arg, ctx)
                return ir.Acosh(arg)
            case UnaryOpKind.ASINH:
                arg = self._visit(e.arg, ctx)
                return ir.Asinh(arg)
            case UnaryOpKind.ATANH:
                arg = self._visit(e.arg, ctx)
                return ir.Atanh(arg)
            case UnaryOpKind.COSH:
                arg = self._visit(e.arg, ctx)
                return ir.Cosh(arg)
            case UnaryOpKind.SINH:
                arg = self._visit(e.arg, ctx)
                return ir.Sinh(arg)
            case UnaryOpKind.TANH:
                arg = self._visit(e.arg, ctx)
                return ir.Tanh(arg)
            case UnaryOpKind.EXP:
                arg = self._visit(e.arg, ctx)
                return ir.Exp(arg)
            case UnaryOpKind.EXP2:
                arg = self._visit(e.arg, ctx)
                return ir.Exp2(arg)
            case UnaryOpKind.EXPM1:
                arg = self._visit(e.arg, ctx)
                return ir.Expm1(arg)
            case UnaryOpKind.LOG:
                arg = self._visit(e.arg, ctx)
                return ir.Log(arg)
            case UnaryOpKind.LOG10:
                arg = self._visit(e.arg, ctx)
                return ir.Log10(arg)
            case UnaryOpKind.LOG1P:
                arg = self._visit(e.arg, ctx)
                return ir.Log1p(arg)
            case UnaryOpKind.LOG2:
                arg = self._visit(e.arg, ctx)
                return ir.Log2(arg)
            case UnaryOpKind.ERF:
                arg = self._visit(e.arg, ctx)
                return ir.Erf(arg)
            case UnaryOpKind.ERFC:
                arg = self._visit(e.arg, ctx)
                return ir.Erfc(arg)
            case UnaryOpKind.LGAMMA:
                arg = self._visit(e.arg, ctx)
                return ir.Lgamma(arg)
            case UnaryOpKind.TGAMMA:
                arg = self._visit(e.arg, ctx)
                return ir.Tgamma(arg)
            case UnaryOpKind.ISFINITE:
                arg = self._visit(e.arg, ctx)
                return ir.IsFinite(arg)
            case UnaryOpKind.ISINF:
                arg = self._visit(e.arg, ctx)
                return ir.IsInf(arg)
            case UnaryOpKind.ISNAN:
                arg = self._visit(e.arg, ctx)
                return ir.IsNan(arg)
            case UnaryOpKind.ISNORMAL:
                arg = self._visit(e.arg, ctx)
                return ir.IsNormal(arg)
            case UnaryOpKind.SIGNBIT:
                arg = self._visit(e.arg, ctx)
                return ir.Signbit(arg)
            case UnaryOpKind.RANGE:
                arg = self._visit(e.arg, ctx)
                return ir.Range(arg)
            case _:
                raise NotImplementedError('unexpected op', e.op)

    def _visit_binaryop(self, e, ctx: _CtxType):
        lhs = self._visit(e.left, ctx)
        rhs = self._visit(e.right, ctx)
        match e.op:
            case BinaryOpKind.ADD:
                return ir.Add(lhs, rhs)
            case BinaryOpKind.SUB:
                return ir.Sub(lhs, rhs)
            case BinaryOpKind.MUL:
                return ir.Mul(lhs, rhs)
            case BinaryOpKind.DIV:
                return ir.Div(lhs, rhs)
            case BinaryOpKind.COPYSIGN:
                return ir.Copysign(lhs, rhs)
            case BinaryOpKind.FDIM:
                return ir.Fdim(lhs, rhs)
            case BinaryOpKind.FMAX:
                return ir.Fmax(lhs, rhs)
            case BinaryOpKind.FMIN:
                return ir.Fmin(lhs, rhs)
            case BinaryOpKind.FMOD:
                return ir.Fmod(lhs, rhs)
            case BinaryOpKind.REMAINDER:
                return ir.Remainder(lhs, rhs)
            case BinaryOpKind.HYPOT:
                return ir.Hypot(lhs, rhs)
            case BinaryOpKind.ATAN2:
                return ir.Atan2(lhs, rhs)
            case BinaryOpKind.POW:
                return ir.Pow(lhs, rhs)
            case _:
                raise NotImplementedError('unexpected op', e.op)

    def _visit_ternaryop(self, e, ctx: _CtxType):
        arg0 = self._visit(e.arg0, ctx)
        arg1 = self._visit(e.arg1, ctx)
        arg2 = self._visit(e.arg2, ctx)
        match e.op:
            case TernaryOpKind.FMA:
                return ir.Fma(arg0, arg1, arg2)
            case TernaryOpKind.DIGITS:
                assert isinstance(arg0, ir.Integer), f'must be an integer, got {arg0}'
                assert isinstance(arg1, ir.Integer), f'must be an integer, got {arg1}'
                assert isinstance(arg2, ir.Integer), f'must be an integer, got {arg2}'
                return ir.Digits(arg0.val, arg1.val, arg2.val)
            case _:
                raise NotImplementedError('unexpected op', e.op)

    def _visit_naryop(self, e, ctx: _CtxType):
        args: list[ir.Expr] = [self._visit(arg, ctx) for arg in e.args]
        match e.op:
            case NaryOpKind.AND:
                return ir.And(*args)
            case NaryOpKind.OR:
                return ir.Or(*args)
            case _:
                raise NotImplementedError('unexpected op', e.op)

    def _visit_compare(self, e, ctx: _CtxType):
        args: list[ir.Expr] = [self._visit(arg, ctx) for arg in e.args]
        return ir.Compare(e.ops, args)

    def _visit_call(self, e, ctx: _CtxType):
        args: list[ir.Expr]  = [self._visit(arg, ctx) for arg in e.args]
        return ir.UnknownCall(e.op, *args)

    def _visit_tuple_expr(self, e, ctx: _CtxType):
        return ir.TupleExpr(*[self._visit(arg, ctx) for arg in e.args])

    def _visit_if_expr(self, e, ctx: _CtxType):
        return ir.IfExpr(
            self._visit(e.cond, ctx),
            self._visit(e.ift, ctx),
            self._visit(e.iff, ctx)
        )

    def _visit_var_assign(self, stmt, ctx: _CtxType):
        # compile the expression
        e = self._visit(stmt.expr, ctx)
        # generate fresh variable for the assignment
        t = self.gensym.fresh(stmt.var)
        ctx = { **ctx, stmt.var: t }
        s = ir.VarAssign(t, ir.AnyType(), e)
        return s, ctx
    
    def _compile_tuple_binding(self, vars: TupleBinding, ctx: _CtxType):
        new_vars: list[str | ir.TupleBinding] = []
        for name in vars:
            if isinstance(name, str):
                new_vars.append(ctx[name])
            elif isinstance(name, TupleBinding):
                new_vars.append(self._compile_tuple_binding(name, ctx))
            else:
                raise NotImplementedError('unexpected tuple identifier', name)
        return ir.TupleBinding(new_vars)

    def _visit_tuple_assign(self, stmt, ctx: _CtxType):
        # compile the expression
        e = self._visit(stmt.expr, ctx)
        # generate fresh variables for the tuple assignment
        for name in stmt.vars.names():
            t = self.gensym.fresh(name)
            ctx = { **ctx, name: t }
        vars = self._compile_tuple_binding(stmt.vars, ctx)
        tys = ir.TensorType([ir.AnyType() for _ in stmt.vars])
        s = ir.TupleAssign(vars, tys, e)
        return s, ctx

    def _visit_if1_stmt(self, stmt: IfStmt, ctx: _CtxType):
        """Like `_visit_if_stmt`, but for 1-armed if statements."""
        cond = self._visit(stmt.cond, ctx)
        body, body_ctx = self._visit_block(stmt.ift, ctx)
        _, live_out = stmt.attribs[LiveVarAnalysis.analysis_name]
        # merge live variables and create new context
        phis: list[ir.PhiNode] = []
        new_ctx: _CtxType = dict()
        for name in live_out:
            old_name = ctx[name]
            new_name = body_ctx[name]
            if old_name != new_name:
                t = self.gensym.fresh(name)
                phis.append(ir.PhiNode(t, old_name, new_name, ir.AnyType()))
                new_ctx[name] = t
            else:
                new_ctx[name] = ctx[name]
        # create new statement
        s = ir.If1Stmt(cond, body, phis)
        return s, new_ctx
    
    def _visit_if2_stmt(self, stmt: IfStmt, ctx: _CtxType):
        """Like `_visit_if_stmt`, but for 2-armed if statements."""
        assert stmt.iff is not None, 'expected a 2-armed if statement'
        cond = self._visit(stmt.cond, ctx)
        ift, ift_ctx = self._visit_block(stmt.ift, ctx)
        iff, iff_ctx = self._visit_block(stmt.iff, ctx)
        _, live_out = stmt.attribs[LiveVarAnalysis.analysis_name]
        # merge live variables
        phis: list[ir.PhiNode] = []
        new_ctx: _CtxType = dict()
        for name in live_out:
            # well-formedness means that the variable is in both contexts
            ift_name = ift_ctx.get(name, None)
            iff_name = iff_ctx.get(name, None)
            assert ift_name is not None, f'variable not in true branch {ift_name}'
            assert iff_name is not None, f'variable not in false branch {iff_name}'
            if ift_name != iff_name:
                # variable updated on at least one branch => create phi node
                t = self.gensym.fresh(name)
                phis.append(ir.PhiNode(t, ift_name, iff_name, ir.AnyType()))
                new_ctx[name] = t
            else:
                # variable not mutated => keep the same name
                new_ctx[name] = ctx[name]
        # create new statement
        s = ir.IfStmt(cond, ift, iff, phis)
        return s, new_ctx

    def _visit_if_stmt(self, stmt, ctx: _CtxType):
        if stmt.iff is None:
            return self._visit_if1_stmt(stmt, ctx)
        else:
            return self._visit_if2_stmt(stmt, ctx)

    def _visit_while_stmt(self, stmt, ctx: _CtxType):
        # merge variables initialized before the block that
        # are updated in the body of the loop
        live_in, live_out = stmt.attribs[LiveVarAnalysis.analysis_name]
        live_cond = stmt.cond.attribs[LiveVarAnalysis.analysis_name]
        live_body, _ = stmt.body.attribs[LiveVarAnalysis.analysis_name]
        _, def_out = stmt.body.attribs[DefinitionAnalysis.analysis_name]
        live_loop: set[str] = live_cond | live_body
        # generate fresh variables for all changed variables
        changed_map: dict[str, str] = dict()
        changed_vars: set[str] = live_in & def_out
        for name in changed_vars:
            t = self.gensym.fresh(name)
            changed_map[name] = t
        # create the new context for the loop
        loop_ctx: _CtxType = dict()
        for name in live_loop:
            if name in changed_map:
                loop_ctx[name] = changed_map[name]
            else:
                loop_ctx[name] = ctx[name]
        # compile the condition and body using the loop context
        cond = self._visit(stmt.cond, loop_ctx)
        body, body_ctx = self._visit_block(stmt.body, loop_ctx)
        # merge all changed variables using phi nodes
        phis: list[ir.PhiNode] = []
        for name, t in changed_map.items():
            old_name = ctx[name]
            new_name = body_ctx[name]
            assert old_name != new_name, 'must be different by definition analysis'
            phis.append(ir.PhiNode(t, old_name, new_name, ir.AnyType()))
        # create new statement and context
        s = ir.WhileStmt(cond, body, phis)
        new_ctx: _CtxType = dict()
        for name in live_out:
            if name in changed_map:
                new_ctx[name] = changed_map[name]
            else:
                new_ctx[name] = ctx[name]
        return s, new_ctx

    def _visit_for_stmt(self, stmt, ctx: _CtxType):
        # compile the iterable expression
        cond = self._visit(stmt.iterable, ctx)
        # generate fresh variable for the loop variable
        iter_var = self.gensym.fresh(stmt.var)
        ctx = { **ctx, stmt.var: iter_var }
        # merge variables initialized before the block that
        # are updated in the body of the loop
        live_in, live_out = stmt.attribs[LiveVarAnalysis.analysis_name]
        live_loop, _ = stmt.body.attribs[LiveVarAnalysis.analysis_name]
        _, def_out = stmt.body.attribs[DefinitionAnalysis.analysis_name]
        # generate fresh variables for all changed variables
        changed_map: dict[str, str] = dict()
        changed_vars: set[str] = live_in & def_out
        for name in changed_vars:
            t = self.gensym.fresh(name)
            changed_map[name] = t
        # create the new context for the loop
        loop_ctx: _CtxType = dict()
        for name in live_loop:
            if name in changed_map:
                loop_ctx[name] = changed_map[name]
            else:
                loop_ctx[name] = ctx[name]
        # compile the loop body using the loop context
        body, body_ctx = self._visit_block(stmt.body, loop_ctx)
        # merge all changed variables using phi nodes
        phis: list[ir.PhiNode] = []
        for name, t in changed_map.items():
            old_name = ctx[name]
            new_name = body_ctx[name]
            assert old_name != new_name, 'must be different by definition analysis'
            phis.append(ir.PhiNode(t, old_name, new_name, ir.AnyType()))
        # create new statement and context
        s = ir.ForStmt(iter_var, ir.AnyType(), cond, body, phis)
        new_ctx: _CtxType = dict()
        for name in live_out:
            if name in changed_map:
                new_ctx[name] = changed_map[name]
            else:
                new_ctx[name] = ctx[name]
        return s, new_ctx

    def _visit_return(self, stmt, ctx: _CtxType):
        e = self._visit(stmt.expr, ctx)
        return ir.Return(e), set()

    def _visit_block(self, block, ctx: _CtxType):
        stmts: list[ir.Stmt] = []
        for stmt in block.stmts:
            new_stmt, ctx = self._visit(stmt, ctx)
            stmts.append(new_stmt)
        return ir.Block(stmts), ctx

    def _visit_function(self, func, ctx: _CtxType):
        ctx = dict(ctx)
        args: list[ir.Argument] = []
        for arg in func.args:
            self.gensym.reserve(arg.name)
            ctx[arg.name] = arg.name
            args.append(ir.Argument(arg.name, ir.AnyType()))
        e, _ = self._visit(func.body, ctx)
        return ir.Function(func.name, args, e, ir.AnyType()) 


class IRCodegen:
    """Lowers a FPy AST to FPy IR."""
    
    @staticmethod
    def lower(f: Function) -> ir.Function:
        return _IRCodegenInstance(f).lower()
