"""Parser for the FPy DSL. Converts a Python AST to an FPy AST"""

import ast
import inspect

from typing import Any, Callable, Optional

from .fpyast import *
from .utils import raise_type_error

class FPyParserError(Exception):
    """Parser error for FPy"""

def _parse_assign_lhs(target: ast.expr, st: ast.stmt):
    match target:
        case ast.Name(id=id):
            return id
        case _:
            raise FPyParserError(f'Expected identifier {target} in {st}')

def _parse_annotation(ann: ast.expr, st: ast.stmt):
    pass

def _parse_unary(e: ast.UnaryOp):
    match e.op:
        case ast.UAdd():
            return _parse_expr(e.operand)
        case ast.USub():
            return Neg(_parse_expr(e.operand))
        case _:
            raise FPyParserError(f'Not a valid FPy operator: {e.op} in {e}')

def _parse_binary(e: ast.BinOp):
    match e.op:
        case ast.Add():
            return Add(_parse_expr(e.left), _parse_expr(e.right))
        case ast.Sub():
            return Sub(_parse_expr(e.left), _parse_expr(e.right))
        case ast.Mult():
            return Mul(_parse_expr(e.left), _parse_expr(e.right))
        case ast.Div():
            return Div(_parse_expr(e.left), _parse_expr(e.right))
        case _:
            raise FPyParserError(f'Not a valid FPy operator: {e.op} in {e}')

def _parse_expr(e: ast.expr):
    match e:
        case ast.UnaryOp():
            return _parse_unary(e)
        case ast.BinOp():
            return _parse_binary(e)
        case ast.Call(func=func, args=args, keywords=keywords):
            print(func)
            raise NotImplementedError
        case _:
            raise FPyParserError(f'Not a valid FPy expression: {e}')

def _parse_statement(st: ast.stmt) -> Stmt:
    match st:
        case ast.AnnAssign(target=target, annotation=ann, value=value):
            if value is None:
                raise FPyParserError(f'Assignment must have a value: {st}')
            name = _parse_assign_lhs(, st)
            return None
        case ast.Assign(targets=targets, value=value):
            match targets:
                case [t0]:
                    name = _parse_assign_lhs(t0, st)
                    return Assign(name, _parse_expr(value))
                case _:
                    raise FPyParserError(f'Unpacking assignment not a valid FPy statement: {st}')
        case ast.Return(value=e):
            if e is None:
                raise FPyParserError(f'Return statement must have value: {st}')
            return Return(_parse_expr(e))
        case _:
            raise FPyParserError(f'Not a valid FPy statement: {st}')

def _parse_function(tree: ast.FunctionDef)-> Function:
    stmts: list[Stmt] = []
    for st in tree.body:
        stmt = _parse_statement(st)
        stmts.append(stmt)
    return Function(name=tree.name, body=stmts)


def parse_tree(tree: ast.FunctionDef):
    """
    Parses an instance of an `ast.FunctionDef` into FPy.
    Valid FPy programs are only a subset of Python programs.
    """
    func = _parse_function(tree)
    return None

def fpcore(*args, **kwargs):
    """
    Constructs a callable (and inspectable) FPY function from
    an arbitrary Python function. If possible, the argument
    is analyzed for type information. The result is either
    the function as-is or a callable `FPCore` instance that
    has additional funcitonality.
    """
    def wrap(func):
        if not callable(func):
            raise_type_error(function, func)

        # re-parse the function and translate it to FPy
        source = inspect.getsource(func)
        ptree = ast.parse(source).body[0]
        expr = parse_tree(ptree)

        raise NotImplementedError(expr)

    match args:
        case []:
            return wrap
        case [func]:
            return wrap(func)
        case _:
            raise TypeError('fpcore() expected only 0 or 1 positional arguments')
