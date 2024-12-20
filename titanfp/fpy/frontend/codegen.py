"""
This module does intermediate code generation, compiling
the abstract syntax tree (AST) to the intermediate representation (IR).
"""

from .definition import DefinitionAnalysis
from .fpyast import *
from .live_vars import LiveVarAnalysis
from .visitor import AstVisitor

from .. import ir
from ..gensym import Gensym

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
        t = self.gensym.fresh(stmt.var)
        ctx = { **ctx, stmt.var: t }
        e = self._visit(stmt.expr, ctx)
        s = ir.VarAssign(t, ir.AnyType(), e)
        return [s], ctx
    
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
        for name in stmt.vars.names():
            t = self.gensym.fresh(name)
            ctx = { **ctx, name: t }
        vars = self._compile_tuple_binding(stmt.vars, ctx)
        e = self._visit(stmt.expr, ctx)
        tys = ir.TensorType([ir.AnyType() for _ in stmt.vars])
        s = ir.TupleAssign(vars, tys, e)
        return [s], ctx

    def _visit_if1_stmt(self, stmt: IfStmt, ctx: _CtxType):
        """Like `_visit_if_stmt`, but for 1-armed if statements."""
        cond = self._visit(stmt.cond, ctx)
        body, body_ctx = self._visit_block(stmt.ift, ctx)
        _, live_out = stmt.attribs[LiveVarAnalysis.analysis_name]
        # merge live variables and create new context
        phis: ir.PhiNodes = dict()
        new_ctx: _CtxType = dict()
        for name in live_out:
            old_name = ctx[name]
            new_name = body_ctx[name]
            if old_name != new_name:
                t = self.gensym.fresh(name)
                phis[t] = (old_name, new_name)
                new_ctx[name] = t
            else:
                new_ctx[name] = ctx[name]
        # create new statement
        s = ir.If1Stmt(cond, body, phis)
        return [s], new_ctx
    
    def _visit_if2_stmt(self, stmt: IfStmt, ctx: _CtxType):
        """Like `_visit_if_stmt`, but for 2-armed if statements."""
        assert stmt.iff is not None, 'expected a 2-armed if statement'
        cond = self._visit(stmt.cond, ctx)
        ift, ift_ctx = self._visit_block(stmt.ift, ctx)
        iff, iff_ctx = self._visit_block(stmt.iff, ctx)
        _, live_out = stmt.attribs[LiveVarAnalysis.analysis_name]
        # merge live variables
        phis: ir.PhiNodes = dict()
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
                phis[t] = (ift_name, iff_name)
                new_ctx[name] = t
            else:
                # variable not mutated => keep the same name
                new_ctx[name] = ctx[name]
        # create new statement
        s = ir.IfStmt(cond, ift, iff, phis)
        return [s], new_ctx

    def _visit_if_stmt(self, stmt, ctx: _CtxType):
        if stmt.iff is None:
            return self._visit_if1_stmt(stmt, ctx)
        else:
            return self._visit_if2_stmt(stmt, ctx)

    def _visit_while_stmt(self, stmt, ctx: _CtxType):
        # merge variables initialized before the block that are updated
        # in the body of the loop
        live_in, live_out = stmt.attribs[LiveVarAnalysis.analysis_name]
        live_cond = stmt.cond.attribs[LiveVarAnalysis.analysis_name]
        live_body, _ = stmt.body.attribs[LiveVarAnalysis.analysis_name]
        _, def_out = stmt.body.attribs[DefinitionAnalysis.analysis_name]
    
        changed_vars: set[str] = live_in & def_out
        live_loop: set[str] = live_cond | live_body
        # generate fresh variables for all changed variables
        changed_map: dict[str, str] = dict()
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
        print(live_in, live_loop, live_out, ctx, loop_ctx)
        # compile the condition and body using the loop context
        cond = self._visit(stmt.cond, loop_ctx)
        body, body_ctx = self._visit_block(stmt.body, loop_ctx)
        # merge all changed variables using phi nodes
        phis: ir.PhiNodes = dict()
        for name, t in changed_map.items():
            old_name = ctx[name]
            new_name = body_ctx[name]
            assert old_name != new_name, 'must be different by definition analysis'
            phis[t] = (old_name, new_name)
        # create new statement and context
        s = ir.WhileStmt(cond, body, phis)
        new_ctx: _CtxType = dict()
        for name in live_out:
            if name in changed_map:
                new_ctx[name] = changed_map[name]
            else:
                new_ctx[name] = ctx[name]
        return [s], new_ctx

    def _visit_for_stmt(self, stmt, ctx: _CtxType):
        # merge variables initialized before the block that are updated
        # in the body of the loop
        # merge variables initialized before the block that are updated
        # in the body of the loop
        live_in, live_out = stmt.attribs[LiveVarAnalysis.analysis_name]
        live_iter = stmt.iterable.attribs[LiveVarAnalysis.analysis_name]
        live_body, _ = stmt.body.attribs[LiveVarAnalysis.analysis_name]
        _, def_out = stmt.body.attribs[DefinitionAnalysis.analysis_name]

        changed_vars: set[str] = live_in & def_out
        live_loop: set[str] = live_iter | live_body
        raise NotImplementedError(changed_vars, live_in, live_body, live_out, ctx)

    def _visit_return(self, stmt, ctx: _CtxType):
        e = self._visit(stmt.expr, ctx)
        s = ir.Return(e)
        return [s], set()

    def _visit_block(self, block, ctx: _CtxType):
        stmts: list[ir.Stmt] = []
        for stmt in block.stmts:
            match stmt:
                case VarAssign():
                    new_stmts, ctx = self._visit(stmt, ctx)
                    stmts.extend(new_stmts)
                case TupleAssign():
                    new_stmts, ctx = self._visit(stmt, ctx)
                    stmts.extend(new_stmts)
                case IfStmt():
                    new_stmts, ctx = self._visit(stmt, ctx)
                    stmts.extend(new_stmts)
                case WhileStmt():
                    new_stmts, ctx = self._visit(stmt, ctx)
                    stmts.extend(new_stmts)
                case ForStmt():
                    new_stmts, ctx = self._visit(stmt, ctx)
                    stmts.extend(new_stmts)
                case Return():
                    new_stmts, ctx = self._visit(stmt, ctx)
                    stmts.extend(new_stmts)
                case _:
                    raise NotImplementedError('unexpected statement', stmt)
        return ir.Block(stmts), ctx

    def _visit_function(self, func, ctx: _CtxType):
        ctx = dict(ctx)
        args: list[ir.Argument] = []
        for arg in func.args:
            ctx[arg.name] = self.gensym.reserve(arg)
            args.append(ir.Argument(arg.name, ir.AnyType()))
        e, _ = self._visit(func.body, ctx)
        return ir.Function(func.name, args, e, ir.AnyType()) 


class IRCodegen:
    """Lowers a FPy AST to FPy IR."""
    
    def lower(self, f: Function) -> ir.Function:
        return _IRCodegenInstance(f).lower()
