"""Parser for the FPy DSL. Converts a Python AST to an FPy AST"""

import ast
import inspect

from functools import reduce
from typing import cast, Callable

from .analysis import SyntaxCheck
from .fpyast import *
from .ops import op_info, op_is_defined
from .utils import raise_type_error

def _ipow(expr: Expr, n: int):
    assert n >= 0, "must be a non-negative integer"
    if n == 0:
        return Integer(1)
    elif n == 1:
        return expr
    else:
        return reduce(Mul, [expr for _ in range(n)])

class FPyParserError(Exception):
    """Parser error for FPy"""
    source: str
    why: str
    where: ast.AST
    ctx: Optional[ast.AST]

    def __init__(self,
        source: str,
        why: str,
        where: ast.AST,
        ctx: Optional[ast.AST] = None,
    ):
        msg_lines = [why]
        match where:
            case ast.expr() | ast.stmt():
                start_line = where.lineno
                start_col = where.col_offset
                msg_lines.append(f' at: {source}:{start_line}:{start_col}')
            case _:
                pass

        msg_lines.append(f' where: {ast.unparse(where)}')
        if ctx is not None:
            msg_lines.append(f' in: {ast.unparse(ctx)}')
        
        super().__init__('\n'.join(msg_lines))
        self.source = source
        self.why = why
        self.where = where
        self.ctx = ctx


class FPyParser:
    """
    Parser for FPy programs from Python ASTs.
    Valid FPy programs are only a subset of Python programs.
    This parser checks for syntactic validity and produces FPy ASTs.
    """

    source: str
    """
    Name of the source location being parsed.
    This is usually the name of the source file that the function resides in.
    Try `inspect.getsourcefile` to retrieve this information.
    """

    strict: bool
    """
    Only use the FPCore-defined set of operators.
    Foreign calls are disallowed.
    This option is off by default.
    """

    def __init__(self, source: str, strict =False):
        self.source = source
        self.strict = strict

    def parse_function(self, func: ast.FunctionDef):
        """
        Parses an instance of an `ast.FunctionDef` into FPy.
        Valid FPy programs are only a subset of Python programs.
        """
        if func.args.vararg:
            raise FPyParserError(self.source, 'FPy does not support variary arguments', func.args.vararg, func)
        if func.args.kwarg:
            raise FPyParserError(self.source, 'FPy does not support keyword arguments', func.args.kwarg, func)

        args: list[Argument] = []
        for arg in func.args.posonlyargs + func.args.args:
            if arg.annotation is None:
                raise FPyParserError(self.source, 'FPy requires argument annotations', arg)
            ann =  self._parse_annotation(arg.annotation, arg)
            args.append(Argument('_' if arg.arg is None else arg.arg, ann))
        
        block = self._parse_statements(func.body)
        return Function(args, block, ident=func.name)

    def _parse_statements(self, stmts: list[ast.stmt]):
        assert stmts != []
        return Block([self._parse_statement(st) for st in stmts])

    def _parse_statement(self, st: ast.stmt) -> Stmt:
        match st:
            case ast.AnnAssign(target=target, annotation=ann, value=value):
                if value is None:
                    raise FPyParserError(self.source, 'Assignment must have a value', st)
                name = self._parse_assign_lhs(target, st)
                ty_ann = self._parse_annotation(ann, st)
                return Assign(name, self._parse_expr(value), ty_ann)
            case ast.Assign(targets=targets, value=value):
                match targets:
                    case []:
                        raise FPyParserError(self.source, 'Unexpected assignment', st)
                    case [t0]:
                        binding = self._parse_assign_lhs(t0, st)
                        match binding:
                            case VarBinding():
                                return Assign(binding, self._parse_expr(value))
                            case TupleBinding():
                                return TupleAssign(binding, self._parse_expr(value))
                            case _:
                                raise FPyParserError(self.source, 'Unexpected binding type', st)
                    case ts:
                        raise FPyParserError(self.source, 'Multiple assignments unsupported', st)
            case ast.If(test=test, body=body, orelse=orelse):
                cond = self._parse_expr(test)
                ift = self._parse_statements(body)
                iff = self._parse_statements(orelse)
                return IfStmt(cond, ift, iff)
            case ast.Return(value=e):
                if e is None:
                    raise FPyParserError(self.source, 'Return statement must have value', st)
                return Return(self._parse_expr(e))
            case _:
                raise FPyParserError(self.source, 'Not a valid FPy statement', st)

    def _parse_digits(self, e: ast.expr, args: list[ast.expr]):
        if len(args) != 3:
            raise FPyParserError(self.source, f'`digits` expects 3 arguments, given {len(args)}', e)

        m = self._parse_expr(args[0])
        if not isinstance(m, Integer):
            raise FPyParserError(self.source, f'first argument of `digits` must be an integer, given {len(args)}', e)

        exp = self._parse_expr(args[1])
        if not isinstance(exp, Integer):
            raise FPyParserError(self.source, f'second argument of `digits` must be an integer', e)
    
        base = self._parse_expr(args[2])
        if not isinstance(base, Integer):
            raise FPyParserError(self.source, f'third argument of `digits` must be an integer', e)
        if base.val < 2:
            raise FPyParserError(self.source, f'third argument of `digits` must greater than 1', e)

        return Digits(m.val, exp.val, base.val)

    def _parse_expr(self, e: ast.expr):
        match e:
            case ast.IfExp(test=cond, body=ift, orelse=iff):
                return IfExpr(self._parse_expr(cond), self._parse_expr(ift), self._parse_expr(iff))
            case ast.UnaryOp():
                return self._parse_unaryop(e)
            case ast.BinOp():
                return self._parse_binop(e)
            case ast.Compare():
                return self._parse_compare(e)
            case ast.Call(func=func, args=args, keywords=keywords):
                for arg in args:
                    if isinstance(arg, ast.Starred):
                        raise FPyParserError(self.source, 'FPy does not support argument unpacking', e)

                if keywords != []:
                    raise FPyParserError(self.source, 'FPy does not support keyword arguments', e)

                name = self._parse_call(func, e)
                if name == 'digits':
                    # special case: digits
                    return self._parse_digits(e, args)
                elif op_is_defined(name):
                    # reserved FPy operator
                    info = op_info(name)
                    assert info is not None, "unexpectedly None"
                    if len(args) != info.arity:
                        raise FPyParserError(self.source, f'`{name}` expects {info.arity} argument, given {len(args)}', e)
                    return info.fpy(*[self._parse_expr(c) for c in args])
                else:
                    # not a defined operator
                    if self.strict:
                        raise FPyParserError(self.source, 'FPy does not allow foreign calls in strict mode', e, func)

                    children = list(map(self._parse_expr, args))
                    return UnknownCall(name, *children)
            case ast.Tuple(elts=elts):
                return Array(*map(self._parse_expr, elts))
            case ast.Constant(value=v):
                match v:
                    case int():
                        return Integer(v)
                    case float():
                        if v.is_integer():
                            return Integer(int(v))
                        else: # TODO: inf or nan
                            return Decnum(str(v))
                    case _:
                        raise FPyParserError(self.source, 'Unsupported constant', e)
            case ast.Name(id=id):
                return Var(id)
            case _:
                raise FPyParserError(self.source, 'Not a valid FPy expression', e)

    def _parse_unaryop(self, e: ast.UnaryOp):
        match e.op:
            case ast.UAdd():
                return cast(Expr, self._parse_expr(e.operand))
            case ast.USub():
                match self._parse_expr(e.operand):
                    case Integer(val=val):
                        return Integer(-val)
                    case n:
                        return Neg(n)
            case _:
                raise FPyParserError(self.source, 'Not a valid FPy operator', e.op, e)

    def _parse_cmpop(self, op: ast.cmpop, e: ast.Compare):
        match op:
            case ast.Lt():
                return CompareOp.LT
            case ast.LtE():
                return CompareOp.LE
            case ast.GtE():
                return CompareOp.GE
            case ast.Gt():
                return CompareOp.GT
            case ast.Eq():
                return CompareOp.EQ
            case ast.NotEq():
                return CompareOp.NE
            case _:
                raise FPyParserError(self.source, 'Not a valid FPy comparator', op, e)

    def _parse_compare(self, e: ast.Compare):
        # comparators are guaranteed to have 2 or more values
        ops = [self._parse_cmpop(op, e) for op in e.ops]
        args = [self._parse_expr(e) for e in [e.left, *e.comparators]]
        return Compare(ops, args)

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
            case ast.Pow():
                base = self._parse_expr(e.left)
                exp = self._parse_expr(e.right)
                if not isinstance(exp, Integer) or exp.val < 0:
                    raise FPyParserError(self.source, 'FPy only supports `**` for small integer exponent, use `pow()` instead', e.op, e)
                return _ipow(base, exp.val)
            case _:
                raise FPyParserError(self.source, 'Not a valid FPy operator', e.op, e)

    def _parse_call(self, call: ast.expr, e: ast.expr):
        match call:
            case ast.Name(id=id):
                return id
            case _:
                raise FPyParserError(self.source, 'FPy application must be an identifier', call, e)

    def _parse_assign_lhs(self, target: ast.expr, st: ast.stmt):
        match target:
            case ast.Name(id=id):
                return VarBinding(id)
            case ast.Tuple(elts=elts):
                bindings = [self._parse_assign_lhs(elt, st) for elt in elts]
                return TupleBinding(*bindings)
            case _:
                raise FPyParserError(self.source, 'FPy expects an identifier', target, st)

    def _parse_annotation(self, ann: ast.expr, st: ast.AST):
        match ann:
            case ast.Name('Real'):
                return RealType()
            case _:
                raise FPyParserError(self.source, 'Unsupported FPy type annotation', ann, st)

def _get_source(func: Callable):
    """Reparses a function returning the relevant lines and source name."""
    sourcename = inspect.getsourcefile(func)
    sourcelines, start_line = inspect.getsourcelines(func)
    empty_lines = '\n'.join(['' for _ in range(1, start_line)])
    source = empty_lines + '\n' + ''.join(sourcelines)
    return source, sourcename


def fpcore(*args, **kwargs):
    """
    Constructs a callable (and inspectable) FPY function from
    an arbitrary Python function. If possible, the argument
    is analyzed for type information. The result is either
    the function as-is or an `FPCore`.
    """
    def wrap(func):
        if not callable(func):
            raise_type_error(Callable, func)

        # re-parse the function and translate it to FPy
        source, sourcename = _get_source(func)
        ptree = ast.parse(source).body[0]
        assert isinstance(ptree, ast.FunctionDef)

        strict = kwargs.get('strict', False)
        parser = FPyParser(sourcename, strict=strict)
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
        scheck = SyntaxCheck()
        scheck.visit(core)

        return core

    match args:
        case []:
            return wrap
        case [func]:
            return wrap(func)
        case _:
            raise TypeError('fpcore() expected only 0 or 1 positional arguments')
