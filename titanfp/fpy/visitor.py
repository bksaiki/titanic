"""Visitor for FPy ASTs"""

from abc import ABC, abstractmethod
from typing import Any

from .fpyast import *
from .ops import op_info

class BaseVisitor(ABC):
    """Visitor base class for FPy programs"""

    #######################################################
    # Expressions

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
    def _visit_variable(self, e: Var, ctx: Any):
        """Visitor method for `Var` nodes."""
        raise NotImplementedError('virtual method')
    
    @abstractmethod
    def _visit_array(self, e: Array, ctx: Any):
        """Visitor method for `ArrayExpr` nodes."""
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
    def _visit_if_expr(self, e: IfExpr, ctx: Any):
        """Visitor method for `IfExpr` nodes."""
        raise NotImplementedError('virtual method')

    def _visit_expr(self, e: Expr, ctx: Any):
        """Dynamic dispatch for all `Expr` nodes."""
        match e:
            case Decnum():
                return self._visit_decnum(e, ctx)
            case Integer():
                return self._visit_integer(e, ctx)
            case Digits():
                return self._visit_digits(e, ctx)
            case Var():
                return self._visit_variable(e, ctx)
            case Array():
                return self._visit_array(e, ctx)
            case UnknownCall():
                return self._visit_unknown(e, ctx)
            case NaryExpr():
                return self._visit_nary_expr(e, ctx)
            case Compare():
                return self._visit_compare(e, ctx)
            case IfExpr():
                return self._visit_if_expr(e, ctx)
            case _:
                raise NotImplementedError('no visitor method for', e)

    #######################################################
    # Statements

    @abstractmethod
    def _visit_assign(self, stmt: Assign, ctx: Any):
        """Visitor method for `Assign` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_tuple_assign(self, stmt: TupleAssign, ctx: Any):
        """Visitor method for `TupleAssign` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_return(self, stmt: Return, ctx: Any):
        """Visitor method for `Return` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_if_stmt(self, stmt: IfStmt, ctx: Any):
        """Visitor method for `IfStmt` nodes."""
        raise NotImplementedError('virtual method')

    def _visit_statement(self, stmt: Stmt, ctx: Any):
        """Dynamic dispatch for all statements."""
        match stmt:
            case Assign():
                return self._visit_assign(stmt, ctx)
            case TupleAssign():
                return self._visit_tuple_assign(stmt, ctx)
            case Return():
                return self._visit_return(stmt, ctx)
            case IfStmt():
                return self._visit_if_stmt(stmt, ctx)
            case _:
                raise NotImplementedError('no visitor method for', stmt)

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

    def _visit(self, e: Ast, ctx: Any):
        """Dynamic dispatch for all primary `AST` nodes."""
        match e:
            case Function():
                return self._visit_function(e, ctx)
            case Stmt():
                return self._visit_statement(e, ctx)
            case Expr():
                return self._visit_expr(e, ctx)
            case Block():
                return self._visit_block(e, ctx)
            case _:
                raise NotImplementedError('no visitor method for', e)
            
    #######################################################
    # Entry

    @abstractmethod
    def visit(self, *args, **kwargs):
        raise NotImplementedError('virtual method')

# Derived visitor types

class Visitor(BaseVisitor):
    """Visitor base class for scanning an FPy program."""

class ReduceVisitor(BaseVisitor):
    """Visitor base class for reducing FPy programs to a value."""

class TransformVisitor(BaseVisitor):
    """Visitor base class for transforming FPy programs"""

# Default visitor types

class DefaultTransformVisitor(TransformVisitor):
    """Default transform visitor: identity operation on an FPy program."""

    #######################################################
    # Expressions

    def _visit_decnum(self, e, ctx):
        return Decnum(e.val)
    
    def _visit_integer(self, e, ctx):
        return Integer(e.val)
    
    def _visit_digits(self, e, ctx):
        return Digits(e.m, e.e, e.b)
    
    def _visit_variable(self, e, ctx):
        return Var(e.name)

    def _visit_array(self, e, ctx):
        return Array(*[self._visit(c, ctx) for c in e.children])

    def _visit_unknown(self, e, ctx):
        return UnknownCall(*[self._visit(c, ctx) for c in e.children])

    def _visit_nary_expr(self, e, ctx):
        info = op_info(e.name)
        if info is None:
            raise NotImplementedError('unreachable', e)
        
        match info.arity:
            case 1:
                arg0 = self._visit(e.children[0], ctx)
                return info.fpc(arg0)
            case 2:
                arg0 = self._visit(e.children[0], ctx)
                arg1 = self._visit(e.children[1], ctx)
                return info.fpc(arg0, arg1)
            case 3:
                arg0 = self._visit(e.children[0], ctx)
                arg1 = self._visit(e.children[1], ctx)
                arg2 = self._visit(e.children[2], ctx)
                return info.fpc(arg0, arg1, arg2)
            case None:
                args = [self._visit(c, ctx) for c in e.children]
                return info.fpc(*args)
            case _:
                raise NotImplementedError('unreachable', e)

    def _visit_compare(self, e, ctx):
        ops = [op for op in e.ops]
        children = [self._visit(c, ctx) for c in e.children]
        return Compare(ops, children)
    
    def _visit_if_expr(self, e, ctx):
        cond = self._visit(e.cond, ctx)
        ift = self._visit(e.ift, ctx)
        iff = self._visit(e.iff, ctx)
        return IfExpr(cond, ift, iff)

    #######################################################
    # Statements

    def _copy_binding(self, binding: Binding):
        match binding:
            case VarBinding():
                return VarBinding(binding.name)
            case TupleBinding():
                return TupleBinding(*[self._copy_binding(b) for b in binding.bindings])
            case _:
                raise NotImplementedError('unexpected', binding)
    
    def _visit_assign(self, stmt, ctx):
        return Assign(self._copy_binding(stmt.var), self._visit(stmt.val, ctx), stmt.ann)

    def _visit_tuple_assign(self, stmt, ctx):
        return TupleAssign(self._copy_binding(stmt.binding), self._visit(stmt.val, ctx))

    def _visit_return(self, stmt, ctx):
        return Return(self._visit(stmt.e, ctx))

    def _visit_if_stmt(self, stmt, ctx):
        cond = self._visit(stmt.cond, ctx)
        ift = self._visit(stmt.ift, ctx)
        iff = self._visit(stmt.iff, ctx)
        return IfStmt(cond, ift, iff)

    def _visit_block(self, stmt, ctx):
        return Block([self._visit(s) for s in stmt.stmts])

    #######################################################
    # Function

    def _visit_function(self, func, ctx):
        return Function(
            args=[arg for arg in func.args],
            body=self._visit(func.body, None),
            ctx=Context(func.ctx.props),
            ident=func.ident,
            name=func.name,
            pre=func.pre
        )
