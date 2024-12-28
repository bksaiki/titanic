"""Visitor for FPy ASTs"""

from abc import ABC, abstractmethod
from typing import Any

from .ir import *

class BaseVisitor(ABC):
    """Visitor base class for the FPy IR."""

    #######################################################
    # Expressions

    @abstractmethod
    def _visit_var(self, e: Var, ctx: Any):
        """Visitor method for `Var` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_decnum(self, e: Decnum, ctx: Any):
        """Visitor method for `Decnum` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_integer(self, e: Integer, ctx: Any):
        """Visitor method for `Integer` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_digits(self, e: Digits, ctx: Any):
        """Visitor method for `Digits` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_unknown(self, e: UnknownCall, ctx: Any):
        """Visitor method for `UnknownCall` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_nary_expr(self, e: NaryExpr, ctx: Any):
        """Visitor method for `NaryExpr` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_compare(self, e: Compare, ctx: Any):
        """Visitor method for `Compare` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_tuple_expr(self, e: TupleExpr, ctx: Any):
        """Visitor method for `TupleExpr` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_ref_expr(self, e: RefExpr, ctx: Any):
        """Visitor method for `RefExpr` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_comp_expr(self, e: CompExpr, ctx: Any):
        """Visitor method for `CompExpr` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_if_expr(self, e: IfExpr, ctx: Any):
        """Visitor method for `IfExpr` nodes."""
        raise NotImplementedError('virtual method')

    #######################################################
    # Statements

    @abstractmethod
    def _visit_var_assign(self, stmt: VarAssign, ctx: Any):
        """Visitor method for `VarAssign` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_tuple_assign(self, stmt: TupleAssign, ctx: Any):
        """Visitor method for `TupleAssign` nodes."""
        raise NotImplementedError('virtual method')
    
    @abstractmethod
    def _visit_if1_stmt(self, stmt: If1Stmt, ctx: Any):
        """Visitor method for `If1Stmt` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_if_stmt(self, stmt: IfStmt, ctx: Any):
        """Visitor method for `IfStmt` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_while_stmt(self, stmt: WhileStmt, ctx: Any):
        """Visitor method for `WhileStmt` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_for_stmt(self, stmt: ForStmt, ctx: Any):
        """Visitor method for `ForStmt` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_return(self, stmt: Return, ctx: Any):
        """Visitor method for `Return` nodes."""
        raise NotImplementedError('virtual method')

    #######################################################
    # Block

    @abstractmethod
    def _visit_block(self, block: Block, ctx: Any):
        """Visitor method for a list of `Stmt` nodes."""
        raise NotImplementedError('virtual method')

    #######################################################
    # Functions

    @abstractmethod
    def _visit_function(self, func: Function, ctx: Any):
        """Visitor for `fpyast.Function`."""
        raise NotImplementedError('virtual method')

    #######################################################
    # Dynamic dispatch

    def _visit_expr(self, e: Expr, ctx: Any):
        """Dynamic dispatch for all `Expr` nodes."""
        match e:
            case Var():
                return self._visit_var(e, ctx)
            case Decnum():
                return self._visit_decnum(e, ctx)
            case Integer():
                return self._visit_integer(e, ctx)
            case Digits():
                return self._visit_digits(e, ctx)
            case UnknownCall():
                return self._visit_unknown(e, ctx)
            case NaryExpr():
                return self._visit_nary_expr(e, ctx)
            case Compare():
                return self._visit_compare(e, ctx)
            case TupleExpr():
                return self._visit_tuple_expr(e, ctx)
            case RefExpr():
                return self._visit_ref_expr(e, ctx)
            case CompExpr():
                return self._visit_comp_expr(e, ctx)
            case IfExpr():
                return self._visit_if_expr(e, ctx)
            case _:
                raise NotImplementedError('no visitor method for', e)

    def _visit_statement(self, stmt: Stmt, ctx: Any):
        """Dynamic dispatch for all statements."""
        match stmt:
            case VarAssign():
                return self._visit_var_assign(stmt, ctx)
            case TupleAssign():
                return self._visit_tuple_assign(stmt, ctx)
            case If1Stmt():
                return self._visit_if1_stmt(stmt, ctx)
            case IfStmt():
                return self._visit_if_stmt(stmt, ctx)
            case WhileStmt():
                return self._visit_while_stmt(stmt, ctx)
            case ForStmt():
                return self._visit_for_stmt(stmt, ctx)
            case Return():
                return self._visit_return(stmt, ctx)
            case _:
                raise NotImplementedError('no visitor method for', stmt)

    def _visit(self, e: Expr | Stmt | Block | Function, ctx: Any):
        """Dynamic dispatch for all primary `AST` nodes."""
        match e:
            case Expr():
                return self._visit_expr(e, ctx)
            case Stmt():
                return self._visit_statement(e, ctx)
            case Block():
                return self._visit_block(e, ctx)
            case Function():
                return self._visit_function(e, ctx)
            case _:
                raise NotImplementedError('no visitor method for', e)

# Derived visitor types

class Visitor(BaseVisitor):
    """Visitor base class for analyzing FPy programs."""

class ReduceVisitor(BaseVisitor):
    """Visitor base class for reducing FPy programs to a value."""

class TransformVisitor(BaseVisitor):
    """Visitor base class for transforming FPy programs"""

# Default visitor implementations

class DefaultVisitor(Visitor):
    """Default visitor: visits all nodes without doing anything."""
    def _visit_var(self, e: Var, ctx: Any):
        pass

    def _visit_decnum(self, e: Decnum, ctx: Any):
        pass

    def _visit_integer(self, e: Integer, ctx: Any):
        pass

    def _visit_digits(self, e: Digits, ctx: Any):
        pass

    def _visit_unknown(self, e: UnknownCall, ctx: Any):
        for c in e.children:
            self._visit(c, ctx)

    def _visit_nary_expr(self, e: NaryExpr, ctx: Any):
        for c in e.children:
            self._visit(c, ctx)

    def _visit_compare(self, e: Compare, ctx: Any):
        for c in e.children:
            self._visit(c, ctx)

    def _visit_tuple_expr(self, e: TupleExpr, ctx: Any):
        for c in e.children:
            self._visit(c, ctx)

    def _visit_ref_expr(self, e: RefExpr, ctx: Any):
        self._visit(e.array, ctx)
        for c in e.indices:
            self._visit(c, ctx)

    def _visit_comp_expr(self, e: CompExpr, ctx: Any):
        for iterable in e.iterables:
            self._visit(iterable, ctx)
        self._visit(e.elt, ctx)

    def _visit_if_expr(self, e: IfExpr, ctx: Any):
        self._visit(e.cond, ctx)
        self._visit(e.ift, ctx)
        self._visit(e.iff, ctx)

    def _visit_var_assign(self, stmt: VarAssign, ctx: Any):
        self._visit(stmt.expr, ctx)

    def _visit_tuple_assign(self, stmt: TupleAssign, ctx: Any):
        self._visit(stmt.expr, ctx)

    def _visit_if1_stmt(self, stmt: If1Stmt, ctx: Any):
        self._visit(stmt.cond, ctx)
        self._visit(stmt.body, ctx)

    def _visit_if_stmt(self, stmt: IfStmt, ctx: Any):
        self._visit(stmt.cond, ctx)
        self._visit(stmt.ift, ctx)
        self._visit(stmt.iff, ctx)

    def _visit_while_stmt(self, stmt: WhileStmt, ctx: Any):
        self._visit(stmt.cond, ctx)
        self._visit(stmt.body, ctx)

    def _visit_for_stmt(self, stmt: ForStmt, ctx: Any):
        self._visit(stmt.iterable, ctx)
        self._visit(stmt.body, ctx)

    def _visit_return(self, stmt: Return, ctx: Any):
        self._visit(stmt.expr, ctx)

    def _visit_block(self, block: Block, ctx: Any):
        for stmt in block.stmts:
            self._visit(stmt, ctx)

    def _visit_function(self, func: Function, ctx: Any):
        self._visit(func.body, ctx)


class DefaultTransformVisitor(TransformVisitor):
    """Default transform visitor: identity operation on an FPy program."""

    #######################################################
    # Expressions

    def _visit_var(self, e, ctx):
        return Var(e.name)

    def _visit_decnum(self, e, ctx: Any):
        return Decnum(e.val)
    
    def _visit_integer(self, e, ctx: Any):
        return Integer(e.val)
    
    def _visit_digits(self, e, ctx: Any):
        return Digits(e.m, e.e, e.b)

    def _visit_unknown(self, e, ctx: Any):
        return UnknownCall(*[self._visit(c, ctx) for c in e.children])

    def _visit_nary_expr(self, e, ctx: Any):
        match e:
            case UnaryExpr():
                arg0 = self._visit(e.children[0], ctx)
                return type(e)(arg0)
            case BinaryExpr():
                arg0 = self._visit(e.children[0], ctx)
                arg1 = self._visit(e.children[1], ctx)
                return type(e)(arg0, arg1)
            case TernaryExpr():
                arg0 = self._visit(e.children[0], ctx)
                arg1 = self._visit(e.children[1], ctx)
                arg2 = self._visit(e.children[2], ctx)
                return type(e)(arg0, arg1, arg2)
            case _:
                raise NotImplementedError('unreachable', e)

    def _visit_compare(self, e, ctx: Any):
        ops = [op for op in e.ops]
        children = [self._visit(c, ctx) for c in e.children]
        return Compare(ops, children)
    
    def _visit_tuple_expr(self, e, ctx):
        return TupleExpr(*[self._visit(c, ctx) for c in e.children])

    def _visit_ref_expr(self, e, ctx):
        array = self._visit(e.array, ctx)
        indices = [self._visit(c, ctx) for c in e.indices]
        return RefExpr(array, *indices)

    def _visit_comp_expr(self, e, ctx):
        iterables = [self._visit(iterable, ctx) for iterable in e.iterables]
        elt = self._visit(e.elt, ctx)
        return CompExpr(e.vars, iterables, elt)

    def _visit_if_expr(self, e, ctx: Any):
        cond = self._visit(e.cond, ctx)
        ift = self._visit(e.ift, ctx)
        iff = self._visit(e.iff, ctx)
        return IfExpr(cond, ift, iff)

    #######################################################
    # Statements

    def _visit_var_assign(self, stmt, ctx: Any):
        val = self._visit(stmt.expr, ctx)
        return VarAssign(stmt.var, stmt.ty, val)

    def _copy_tuple_binding(self, binding: TupleBinding):
        new_vars: list[str | TupleBinding] = []
        for elt in binding:
            if isinstance(elt, str):
                new_vars.append(elt)
            elif isinstance(elt, TupleBinding):
                new_vars.append(self._copy_tuple_binding(elt))
            else:
                raise NotImplementedError('unexpected tuple element', elt)
        return TupleBinding(new_vars)

    def _visit_tuple_assign(self, stmt, ctx: Any):
        vars = self._copy_tuple_binding(stmt.binding)
        val = self._visit(stmt.expr, ctx)
        return TupleAssign(vars, stmt.ty, val)

    def _visit_if1_stmt(self, stmt, ctx):
        cond = self._visit(stmt.cond, ctx)
        body = self._visit(stmt.body, ctx)
        phis = [self._visit_phi(phi, ctx) for phi in stmt.phis]
        return If1Stmt(cond, body, phis)

    def _visit_if_stmt(self, stmt, ctx: Any):
        cond = self._visit(stmt.cond, ctx)
        ift = self._visit(stmt.ift, ctx)
        iff = self._visit(stmt.iff, ctx)
        phis = [self._visit_phi(phi, ctx) for phi in stmt.phis]
        return IfStmt(cond, ift, iff, phis)
    
    def _visit_while_stmt(self, stmt, ctx):
        cond = self._visit(stmt.cond, ctx)
        body = self._visit(stmt.body, ctx)
        phis = [self._visit_phi(phi, ctx) for phi in stmt.phis]
        return WhileStmt(cond, body, phis)

    def _visit_for_stmt(self, stmt, ctx):
        iterable = self._visit(stmt.iterable, ctx)
        body = self._visit(stmt.body, ctx)
        phis = [self._visit_phi(phi, ctx) for phi in stmt.phis]
        return ForStmt(stmt.var, stmt.ty, iterable, body, phis)

    def _visit_return(self, stmt, ctx: Any):
        return Return(self._visit(stmt.expr, ctx))

    #######################################################
    # Phi node

    def _visit_phi(self, phi: PhiNode, ctx: Any):
        return PhiNode(phi.name, phi.lhs, phi.rhs, phi.ty)

    #######################################################
    # Block

    def _visit_block(self, block: Block, ctx: Any):
        return Block([self._visit(s, ctx) for s in block.stmts])

    #######################################################
    # Function

    def _visit_function(self, func, ctx: Any):
        return Function(func.name, func.args, self._visit(func.body, ctx), func.ty)
