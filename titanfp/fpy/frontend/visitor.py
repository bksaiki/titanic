"""Visitor for the AST of the FPy language."""

from abc import ABC, abstractmethod
from typing import Any

from .fpyast import *

class AstVisitor(ABC):
    """
    Visitor base class for FPy AST nodes.
    """

    #######################################################
    # Expressions

    @abstractmethod
    def _visit_var(self, node: Var, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_decnum(self, node: Decnum, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_integer(self, node: Integer, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_unaryop(self, node: UnaryOp, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_binaryop(self, node: BinaryOp, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_ternaryop(self, node: TernaryOp, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_naryop(self, node: NaryOp, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_compare(self, node: Compare, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_call(self, node: Call, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_tuple_expr(self, node: TupleExpr, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_if_expr(self, node: IfExpr, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    #######################################################
    # Statements

    @abstractmethod
    def _visit_var_assign(self, node: VarAssign, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_tuple_assign(self, node: TupleAssign, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_if_stmt(self, node: IfStmt, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_while_stmt(self, node: WhileStmt, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_for_stmt(self, node: ForStmt, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_return_stmt(self, node: Return, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    #######################################################
    # Funciton

    @abstractmethod
    def _visit_function(self, node: Function, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    #######################################################
    # Dynamic dispatch

    def _visit_expr(self, node: Expr, ctx: Any) -> Any:
        """Dispatch to the appropriate visit method for an expression node."""
        match node:
            case Var():
                return self._visit_var(node, ctx)
            case Decnum():
                return self._visit_decnum(node, ctx)
            case Integer():
                return self._visit_integer(node, ctx)
            case UnaryOp():
                return self._visit_unaryop(node, ctx)
            case BinaryOp():
                return self._visit_binaryop(node, ctx)
            case TernaryOp():
                return self._visit_ternaryop(node, ctx)
            case NaryOp():
                return self._visit_naryop(node, ctx)
            case Compare():
                return self._visit_compare(node, ctx)
            case Call():
                return self._visit_call(node, ctx)
            case TupleExpr():
                return self._visit_tuple_expr(node, ctx)
            case IfExpr():
                return self._visit_if_expr(node, ctx)
            case _:
                raise NotImplementedError(f'visit_expr: {node}')

    def _visit_statement(self, node: Stmt, ctx: Any) -> Any:
        """Dispatch to the appropriate visit method for a statement node."""
        match node:
            case VarAssign():
                return self._visit_var_assign(node, ctx)
            case TupleAssign():
                return self._visit_tuple_assign(node, ctx)
            case IfStmt():
                return self._visit_if_stmt(node, ctx)
            case WhileStmt():
                return self._visit_while_stmt(node, ctx)
            case ForStmt():
                return self._visit_for_stmt(node, ctx)
            case Return():
                return self._visit_return_stmt(node, ctx)
            case _:
                raise NotImplementedError(f'visit_statement: {node}')

    def _visit(self, e: Ast, ctx: Any) -> Any:
        """Dispatch to the appropriate visit method for an AST node."""
        match e:
            case Expr():
                return self._visit_expr(e, ctx)
            case Stmt():
                return self._visit_statement(e, ctx)
            case Function():
                return self._visit_function(e, ctx)
            case _:
                raise NotImplementedError(f'visit: {e}')

    #######################################################
    # Entry

    @abstractmethod
    def visit(self, *args, **kwargs):
        """Entry point for the visitor."""
        raise NotImplementedError('virtual method')
