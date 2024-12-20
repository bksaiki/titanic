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
    def _visit_var(self, e: Var, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_decnum(self, e: Decnum, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_integer(self, e: Integer, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_unaryop(self, e: UnaryOp, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_binaryop(self, e: BinaryOp, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_ternaryop(self, e: TernaryOp, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_naryop(self, e: NaryOp, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_compare(self, e: Compare, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_call(self, e: Call, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_tuple_expr(self, e: TupleExpr, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_if_expr(self, e: IfExpr, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    #######################################################
    # Statements

    @abstractmethod
    def _visit_var_assign(self, stmt: VarAssign, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_tuple_assign(self, stmt: TupleAssign, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_if_stmt(self, stmt: IfStmt, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_while_stmt(self, stmt: WhileStmt, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_for_stmt(self, stmt: ForStmt, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_return(self, stmt: Return, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    #######################################################
    # Block

    @abstractmethod
    def _visit_block(self, block: Block, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    #######################################################
    # Function

    @abstractmethod
    def _visit_function(self, func: Function, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    #######################################################
    # Dynamic dispatch

    def _visit_expr(self, e: Expr, ctx: Any) -> Any:
        """Dispatch to the appropriate visit method for an expression."""
        match e:
            case Var():
                return self._visit_var(e, ctx)
            case Decnum():
                return self._visit_decnum(e, ctx)
            case Integer():
                return self._visit_integer(e, ctx)
            case UnaryOp():
                return self._visit_unaryop(e, ctx)
            case BinaryOp():
                return self._visit_binaryop(e, ctx)
            case TernaryOp():
                return self._visit_ternaryop(e, ctx)
            case NaryOp():
                return self._visit_naryop(e, ctx)
            case Compare():
                return self._visit_compare(e, ctx)
            case Call():
                return self._visit_call(e, ctx)
            case TupleExpr():
                return self._visit_tuple_expr(e, ctx)
            case IfExpr():
                return self._visit_if_expr(e, ctx)
            case _:
                raise NotImplementedError(f'unreachable {e}')

    def _visit_statement(self, stmt: Stmt, ctx: Any) -> Any:
        """Dispatch to the appropriate visit method for a statement."""
        match stmt:
            case VarAssign():
                return self._visit_var_assign(stmt, ctx)
            case TupleAssign():
                return self._visit_tuple_assign(stmt, ctx)
            case IfStmt():
                return self._visit_if_stmt(stmt, ctx)
            case WhileStmt():
                return self._visit_while_stmt(stmt, ctx)
            case ForStmt():
                return self._visit_for_stmt(stmt, ctx)
            case Return():
                return self._visit_return(stmt, ctx)
            case _:
                raise NotImplementedError(f'unreachable: {stmt}')

    def _visit(self, e: Ast, ctx: Any) -> Any:
        """Dispatch to the appropriate visit method for an AST node."""
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
                raise NotImplementedError(f'no visitor method for {e}')
