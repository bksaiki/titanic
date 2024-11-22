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

class FPyParser:
    """
    Parser for FPy programs from Python ASTs.
    Valid FPy programs are only a subset of Python programs.
    This parser checks for syntactic validity and produces FPy ASTs.
    """

    strict: bool
    """
    Only use the FPCore-defined set of operators.
    Foreign calls are disallowed.
    This option is off by default.
    """

    def __init__(self, strict=False):
        self.strict = strict

    def parse_function(self, func: ast.FunctionDef):
        """
        Parses an instance of an `ast.FunctionDef` into FPy.
        Valid FPy programs are only a subset of Python programs.
        """
        if func.args.vararg:
            raise FPyParserError(f'FPy does not support variary arguments: {func.args.vararg} in {func}')
        if func.args.kwarg:
            raise FPyParserError(f'FPy does not support keyword arguments: {func.args.kwarg} in {func}')

        args: list[Argument] = []
        for arg in func.args.posonlyargs + func.args.args:
            if arg.annotation is None:
                raise FPyParserError(f'FPy requires argument annotations {arg.arg}')
            ann =  self._parse_annotation(arg.annotation, arg)
            args.append(Argument('_' if arg.arg is None else arg.arg, ann))
        
        block = self._parse_statements(func.body)
        return Function(args, block, ident=func.name)

    def _parse_statements(self, sts: list[ast.stmt]):
        match sts:
            case []:
                raise FPyParserError(f'Missing body statements: {sts}')
            case [st]:
                return self._parse_statement(st)
            case _:
                stmts: list[Stmt] = []
                for st in sts:
                    stmts.append(self._parse_statement(st))
                return Block(stmts)

    def _parse_statement(self, st: ast.stmt) -> Stmt:
        match st:
            case ast.AnnAssign(target=target, annotation=ann, value=value):
                if value is None:
                    raise FPyParserError(f'Assignment must have a value: {st}')
                name = self._parse_assign_lhs(target, st)
                ty_ann = self._parse_annotation(ann, st)
                return Assign(name, self._parse_expr(value), ty_ann)
            case ast.Assign(targets=targets, value=value):
                match targets:
                    case [t0]:
                        name = self._parse_assign_lhs(t0, st)
                        return Assign(name, self._parse_expr(value))
                    case _:
                        raise FPyParserError(f'Unpacking assignment not a valid FPy statement: {st}')
            case ast.Return(value=e):
                if e is None:
                    raise FPyParserError(f'Return statement must have value: {st}')
                return Return(self._parse_expr(e))
            case _:
                raise FPyParserError(f'Not a valid FPy statement: {st}')

    def _parse_expr(self, e: ast.expr):
        match e:
            case ast.UnaryOp():
                return self._parse_unaryop(e)
            case ast.BinOp():
                return self._parse_binop(e)
            case ast.Call(func=func, args=args, keywords=keywords):
                for arg in args:
                    if isinstance(arg, ast.Starred):
                        raise FPyParserError(f'FPy does not support argument unpacking: {e}')

                if keywords != []:
                    raise FPyParserError(f'FPy does not support keyword arguments: {e}')

                name = self._parse_call(func, e)
                match _call_table.get(name, None):
                    case None:
                        # not a defined operator
                        if self.strict:
                            raise FPyParserError(f'FPy does not allow foreign calls in strict mode: {e}')

                        children = list(map(self._parse_expr, args))
                        return UnknownCall(name, *children)
                    case (1, cls):
                        # defined unary operator
                        if len(args) != 1:
                            raise FPyParserError(f'FPy operator {name} expects 1 argument, given {len(args)} at {e}')
                        return cls(self._parse_expr(args[0]))
                    case _:
                        raise NotImplementedError('call', name)
            case ast.Constant(value=v):
                match v:
                    case int() | float():
                        return Num(v)
                    case _:
                        raise FPyParserError(f'Unsupported constant: {e}') 
            case ast.Name(id=id):
                return Var(id)
            case _:
                raise FPyParserError(f'Not a valid FPy expression: {e}')

    def _parse_unaryop(self, e: ast.UnaryOp):
        match e.op:
            case ast.UAdd():
                return self._parse_expr(e.operand)
            case ast.USub():
                return Neg(self._parse_expr(e.operand))
            case _:
                raise FPyParserError(f'Not a valid FPy operator: {e.op} in {e}')

    def _parse_binop(self, e: ast.BinOp):
        match e.op:
            case ast.Add():
                return Add(self._parse_expr(e.left), self._parse_expr(e.right))
            case ast.Sub():
                return Sub(self._parse_expr(e.left), self._parse_expr(e.right))
            case ast.Mult():
                return Mul(self._parse_expr(e.left), self._parse_expr(e.right))
            case ast.Div():
                return Div(self._parse_expr(e.left), self._parse_expr(e.right))
            case _:
                raise FPyParserError(f'Not a valid FPy operator: {e.op} in {e}')

    def _parse_call(self, call: ast.expr, e: ast.expr):
        match call:
            case ast.Name(id=id):
                return id
            case _:
                raise FPyParserError(f'FPy application must be an identifier: {call} in {e}')

    def _parse_assign_lhs(self, target: ast.expr, st: ast.stmt):
        match target:
            case ast.Name(id=id):
                return id
            case _:
                raise FPyParserError(f'FPy expects an identifier {target} in {st}')

    def _parse_annotation(self, ann: ast.expr, st: ast.AST):
        match ann:
            case ast.Name('Real'):
                return RealType()
            case _:
                raise FPyParserError(f'Unsupported FPy type annotation {ann} in {st}')


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
            raise_type_error(Callable, func)

        # re-parse the function and translate it to FPy
        source = inspect.getsource(func)
        ptree = ast.parse(source).body[0]
        assert isinstance(ptree, ast.FunctionDef)

        strict = kwargs.get('strict', False)
        parser = FPyParser(strict=strict)
        core = parser.parse_function(ptree)

        # handle keywords
        for k in kwargs:
            if k == 'name':
                core.name = kwargs[k]
            elif k == 'pre':
                decorators = ptree.decorator_list
                raise NotImplementedError('precondition')
                # pre = kwargs[k]
                # if isinstance(pre, Callable):
                #     source = inspect.getsource(pre).replace('pre=', '')
                #     print(source)
                #     ptree = ast.parse(source, mode='eval')
                #     raise NotImplementedError(ptree)
                # else:
                #     raise FPyParserError(f'invalid precondition, expected a function {pre}')
            else:
                # TODO: check for structured data
                pass

        core.ctx = Context(props=kwargs)

        # TODO: static analysis:
        #  - unknown variables
        #  - type checking

        return core

    match args:
        case []:
            return wrap
        case [func]:
            return wrap(func)
        case _:
            raise TypeError('fpcore() expected only 0 or 1 positional arguments')
