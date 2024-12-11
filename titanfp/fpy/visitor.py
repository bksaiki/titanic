"""Visitor for FPy ASTs"""

from abc import ABC, abstractmethod
from typing import Any

from .fpyast import *

class BaseVisitor(ABC):
    """Visitor base class for FPy programs"""

    #######################################################
    # Expressions

    @abstractmethod
    def _visit_number(self, e: Num, ctx: Any):
        """Visitor method for `Num` nodes."""
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
    def _visit_array(self, e: ArrayExpr, ctx: Any):
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
            case Num():
                return self._visit_number(e, ctx)
            case Digits():
                return self._visit_digits(e, ctx)
            case Var():
                return self._visit_variable(e, ctx)
            case ArrayExpr():
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
                raise NotImplementedError('visitor method not found for', e)

    #######################################################
    # Statements

    @abstractmethod
    def _visit_assign(self, stmt: Assign, stmts: list[Stmt], ctx: Any):
        """Visitor method for `Assign` nodes."""
        raise NotImplementedError('virtual method')
    
    @abstractmethod
    def _visit_tuple_assign(self, stmt: TupleAssign, stmts: list[Stmt], ctx: Any):
        """Visitor method for `TupleAssign` nodes."""
        raise NotImplementedError('virtual method')
    
    @abstractmethod
    def _visit_return(self, stmt: Return, stmts: list[Stmt], ctx: Any):
        """Visitor method for `Return` nodes."""
        raise NotImplementedError('virtual method')
    
    def _visit_statements(self, stmts: list[Stmt], ctx: Any):
        """Default visitor method for a list of `Stmt` nodes."""
        match stmts:
            case []:
                raise ValueError('Cannot visit an empty list of statements')
            case [Assign() as st, *rest]:
                return self._visit_assign(st, rest, ctx)
            case [TupleAssign() as st, *rest]:
                return self._visit_tuple_assign(st, rest, ctx)
            case [Return() as st, *rest]:
                return self._visit_return(st, rest, ctx)
            case [st, *_]:
                raise NotImplementedError('visitor method not found for', st)
            case _:
                raise NotImplementedError('unreachable')
    
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
                return self._visit_statements([e], ctx)
            case Expr():
                return self._visit_expr(e, ctx)
            case _:
                raise NotImplementedError('visitor method not found for', e)


class ReduceVisitor(BaseVisitor):
    """Visitor base class for reducing FPy programs to a value."""
    pass
