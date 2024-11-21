"""Parser for the FPy DSL. Converts a Python AST to an FPy AST"""

import ast
import inspect

from typing import cast, Callable, Type

from .fpyast import *
from .utils import raise_type_error

_call_table: dict[str, tuple[int, Callable[..., Expr]]] = {
    'fabs' : (1, Fabs),
    'sqrt' : (1, Sqrt)
}

class FPyParserError(Exception):
    """Parser error for FPy"""

def _parse_assign_lhs(target: ast.expr, st: ast.stmt):
    match target:
        case ast.Name(id=id):
            return id
        case _:
            raise FPyParserError(f'FPy expects an identifier {target} in {st}')

def _parse_annotation(ann: ast.expr, st: ast.stmt):
    match ann:
        case ast.Name('Real'):
            return RealType()
        case _:
            raise FPyParserError(f'Unsupported FPy type annotation {ann} in {st}')
            
    print(ann)
    pass

def _parse_unaryop(e: ast.UnaryOp):
    match e.op:
        case ast.UAdd():
            return _parse_expr(e.operand)
        case ast.USub():
            return Neg(_parse_expr(e.operand))
        case _:
            raise FPyParserError(f'Not a valid FPy operator: {e.op} in {e}')

def _parse_binop(e: ast.BinOp):
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

def _parse_call(call: ast.expr, e: ast.expr):
    match call:
        case ast.Name(id=id):
            return id
        case _:
            raise FPyParserError(f'FPy application must be an identifier: {call} in {e}')

def _parse_expr(e: ast.expr):
    match e:
        case ast.UnaryOp():
            return _parse_unaryop(e)
        case ast.BinOp():
            return _parse_binop(e)
        case ast.Call(func=func, args=args, keywords=keywords):
            for arg in args:
                if isinstance(arg, ast.Starred):
                    raise FPyParserError(f'FPy does not support argument unpacking: {e}')

            if keywords != []:
                raise FPyParserError(f'FPy does not support keyword arguments: {e}')

            name = _parse_call(func, e)
            match _call_table.get(name, None):
                case None:
                    # not a defined operator
                    children = list(map(_parse_expr, args))
                    return UnknownCall(name, *children)
                case (1, cls):
                    # defined unary operator
                    if len(args) != 1:
                        raise FPyParserError(f'FPy operator {name} expects 1 argument, given {len(args)} at {e}')
                    return cls(_parse_expr(args[0]))
                case _:
                    raise NotImplementedError('call', name)
        case ast.Constant(value=v):
            match v:
                case int():
                    return Real(v)
                case _:
                    raise FPyParserError(f'Unsupported constant: {e}') 
        case ast.Name(id=id):
            return Var(id)
        case _:
            raise FPyParserError(f'Not a valid FPy expression: {e}')

def _parse_statement(st: ast.stmt) -> Stmt:
    match st:
        case ast.AnnAssign(target=target, annotation=ann, value=value):
            if value is None:
                raise FPyParserError(f'Assignment must have a value: {st}')
            name = _parse_assign_lhs(target, st)
            ty_ann = _parse_annotation(ann, st)
            return Assign(name, ty_ann, _parse_expr(value))
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

def _parse_statements(sts: list[ast.stmt]):
    match sts:
        case []:
            raise FPyParserError(f'Missing body statements: {sts}')
        case [st]:
            return _parse_statement(st)
        case _:
            stmts: list[Stmt] = []
            for st in sts:
                stmts.append(_parse_statement(st))
            return Block(stmts)

def _parse_function(tree: ast.FunctionDef)-> Function:
    block = _parse_statements(tree.body)
    return Function(name=tree.name, body=block)


def parse_tree(tree: ast.FunctionDef):
    """
    Parses an instance of an `ast.FunctionDef` into FPy.
    Valid FPy programs are only a subset of Python programs.
    """
    func = _parse_function(tree)
    return func

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

        # TODO: static analysis:
        #  - unknown variables
        #  - type checking

        return expr

    match args:
        case []:
            return wrap
        case [func]:
            return wrap(func)
        case _:
            raise TypeError('fpcore() expected only 0 or 1 positional arguments')
