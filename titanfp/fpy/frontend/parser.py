"""
This module contains the parser for the FPy language.
"""

import ast

from functools import reduce
from typing import cast

from .fpyast import *

def _ipow(expr: Expr, n: int, loc: Location):
    assert n >= 0, "must be a non-negative integer"
    if n == 0:
        return Integer(1, loc)
    elif n == 1:
        return expr
    else:
        f = lambda lhs, rhs: BinaryOp(BinaryOpKind.MUL, lhs, rhs, loc)
        return reduce(f, [expr for _ in range(n)])

_unary_table = {
    'fabs': UnaryOpKind.FABS,
    'sqrt': UnaryOpKind.SQRT,
    'cbrt': UnaryOpKind.CBRT,
    'ceil': UnaryOpKind.CEIL,
    'floor': UnaryOpKind.FLOOR,
    'nearbyint': UnaryOpKind.NEARBYINT,
    'round': UnaryOpKind.ROUND,
    'trunc': UnaryOpKind.TRUNC,
    'acos': UnaryOpKind.ACOS,
    'asin': UnaryOpKind.ASIN,
    'atan': UnaryOpKind.ATAN,
    'cos': UnaryOpKind.COS,
    'sin': UnaryOpKind.SIN,
    'tan': UnaryOpKind.TAN,
    'acosh': UnaryOpKind.ACOSH,
    'asinh': UnaryOpKind.ASINH,
    'atanh': UnaryOpKind.ATANH,
    'cosh': UnaryOpKind.COSH,
    'sinh': UnaryOpKind.SINH,
    'tanh': UnaryOpKind.TANH,
    'exp': UnaryOpKind.EXP,
    'exp2': UnaryOpKind.EXP2,
    'expm1': UnaryOpKind.EXPM1,
    'log': UnaryOpKind.LOG,
    'log10': UnaryOpKind.LOG10,
    'log1p': UnaryOpKind.LOG1P,
    'log2': UnaryOpKind.LOG2,
    'erf': UnaryOpKind.ERF,
    'erfc': UnaryOpKind.ERFC,
    'lgamma': UnaryOpKind.LGAMMA,
    'tgamma': UnaryOpKind.TGAMMA,
    'isfinite': UnaryOpKind.ISFINITE,
    'isinf': UnaryOpKind.ISINF,
    'isnan': UnaryOpKind.ISNAN,
    'isnormal': UnaryOpKind.ISNORMAL,
    'signbit': UnaryOpKind.SIGNBIT,
    'not': UnaryOpKind.NOT,
    'range': UnaryOpKind.RANGE
}

_binary_table = {
    'add': BinaryOpKind.ADD,
    'sub': BinaryOpKind.SUB,
    'mul': BinaryOpKind.MUL,
    'div': BinaryOpKind.DIV,
    'copysign': BinaryOpKind.COPYSIGN,
    'fdim': BinaryOpKind.FDIM,
    'fmax': BinaryOpKind.FMAX,
    'fmin': BinaryOpKind.FMIN,
    'fmod': BinaryOpKind.FMOD,
    'remainder': BinaryOpKind.REMAINDER,
    'hypot': BinaryOpKind.HYPOT,
    'atan2': BinaryOpKind.ATAN2,
    'pow': BinaryOpKind.POW
}

_ternary_table = {
    'fma': TernaryOpKind.FMA,
    'digits': TernaryOpKind.DIGITS
}

class FPyParserError(Exception):
    """Parser error for FPy"""
    loc: Location
    why: str
    where: ast.AST
    ctx: Optional[ast.AST]

    def __init__(
        self,
        loc: Location,
        why: str,
        where: ast.AST,
        ctx: Optional[ast.AST] = None
    ):
        msg_lines = [why]
        match where:
            case ast.expr() | ast.stmt():
                start_line = loc.start_line
                start_col = loc.start_column
                msg_lines.append(f' at: {loc.source}:{start_line}:{start_col}')
            case _:
                pass

        msg_lines.append(f' where: {ast.unparse(where)}')
        if ctx is not None:
            msg_lines.append(f' in: {ast.unparse(ctx)}')

        super().__init__('\n'.join(msg_lines))
        self.loc = loc
        self.why = why
        self.where = where
        self.ctx = ctx


class Parser:
    """
    FPy parser.
    
    Converts a Python AST (from the `ast` module) to a FPy AST.
    """
    
    name: str
    source: str
    lines: list[str]
    start_line: int

    def __init__(
        self,
        name: str, 
        source: str,
        start_line: int = 1
    ):
        self.name = name
        self.source = source
        self.lines = source.splitlines()
        self.start_line = start_line

    def parse(self):
        """
        Parse the source code into an FPy AST.
        """
        mod = ast.parse(self.source, self.name)
        assert isinstance(mod, ast.Module), "expected module"
        # just grab the first function
        ptree = mod.body[0]
        match ptree:
            case ast.FunctionDef():
                return self._parse_function(ptree)
            case ast.expr():
                return self._parse_expr(ptree)
            case ast.stmt():
                return self._parse_statement(ptree)
            case _:
                raise NotImplementedError('cannot parse', ptree)

    def _parse_location(self, e: ast.expr | ast.stmt) -> Location:
        """Extracts the parse location of a  Python ST node."""
        assert e.end_lineno is not None, "missing end line number"
        assert e.end_col_offset is not None, "missing end column offset"
        return Location(
            self.name,
            e.lineno + self.start_line - 1,
            e.col_offset,
            e.end_lineno + self.start_line - 1,
            e.end_col_offset
        )


    def _parse_type_annotation(self, ann: ast.expr) -> TypeAnn:
        loc = self._parse_location(ann)
        match ann:
            case ast.Name('Real'):
                return ScalarTypeAnn(ScalarType.REAL, loc)
            case ast.Name('int'):
                return ScalarTypeAnn(ScalarType.REAL, loc)
            case ast.Name('float'):
                return ScalarTypeAnn(ScalarType.REAL, loc)
            case ast.Name('bool'):
                return ScalarTypeAnn(ScalarType.BOOL, loc)
            case _:
                raise FPyParserError(loc, 'Unsupported FPy type annotation', ann)

    def _parse_unaryop(self, e: ast.UnaryOp):
        loc = self._parse_location(e)
        match e.op:
            case ast.UAdd():
                arg = self._parse_expr(e.operand)
                return cast(Expr, arg)
            case ast.USub():
                arg = self._parse_expr(e.operand)
                if isinstance(arg, Integer):
                    return Integer(-arg.val, loc)
                else:
                    return UnaryOp(UnaryOpKind.NEG, arg, loc)
            case ast.Not():
                arg = self._parse_expr(e.operand)
                return UnaryOp(UnaryOpKind.NOT, arg, loc)
            case _:
                raise FPyParserError(loc, 'Not a valid FPy operator', e.op, e)

    def _parse_binop(self, e: ast.BinOp):
        loc = self._parse_location(e)
        match e.op:
            case ast.Add():
                lhs = self._parse_expr(e.left)
                rhs = self._parse_expr(e.right)
                return BinaryOp(BinaryOpKind.ADD, lhs, rhs, loc)
            case ast.Sub():
                lhs = self._parse_expr(e.left)
                rhs = self._parse_expr(e.right)
                return BinaryOp(BinaryOpKind.SUB, lhs, rhs, loc)
            case ast.Mult():
                lhs = self._parse_expr(e.left)
                rhs = self._parse_expr(e.right)
                return BinaryOp(BinaryOpKind.MUL, lhs, rhs, loc)
            case ast.Div():
                lhs = self._parse_expr(e.left)
                rhs = self._parse_expr(e.right)
                return BinaryOp(BinaryOpKind.DIV, lhs, rhs, loc)
            case ast.Pow():
                base = self._parse_expr(e.left)
                exp = self._parse_expr(e.right)
                if not isinstance(exp, Integer) or exp.val < 0:
                    raise FPyParserError(loc, 'FPy only supports `**` for small integer exponent, use `pow()` instead', e.op, e)
                return _ipow(base, exp.val, loc)
            case _:
                raise FPyParserError(loc, 'Not a valid FPy operator', e.op, e)

    def _parse_cmpop(self, op: ast.cmpop, e: ast.Compare):
        loc = self._parse_location(e)
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
                raise FPyParserError(loc, 'Not a valid FPy comparator', op, e)

    def _parse_compare(self, e: ast.Compare):
        loc = self._parse_location(e)
        ops = [self._parse_cmpop(op, e) for op in e.ops]
        args = [self._parse_expr(e) for e in [e.left, *e.comparators]]
        return Compare(ops, args, loc)

    def _parse_call(self, e: ast.Call) -> str:
        """Parse a Python call expression."""
        loc = self._parse_location(e)
        if not isinstance(e.func, ast.Name):
            raise FPyParserError(loc, 'Unsupported call expression', e)
        return e.func.id

    def _parse_expr(self, e: ast.expr) -> Expr:
        """Parse a Python expression."""
        loc = self._parse_location(e)
        match e:
            case ast.Name():
                return Var(e.id, loc)
            case ast.Constant():
                # TODO: reparse all constants to get exact value
                if isinstance(e.value, int):
                    return Integer(e.value, loc)
                elif isinstance(e.value, float):
                    if e.value.is_integer():
                        return Integer(int(e.value), loc)
                    else:
                        return Decnum(str(e.value), loc)
                else:
                    raise FPyParserError(loc, 'Unsupported constant', e)
            case ast.UnaryOp():
                return self._parse_unaryop(e)
            case ast.BinOp():
                return self._parse_binop(e)
            case ast.Compare():
                return self._parse_compare(e)
            case ast.Call():
                name = self._parse_call(e)
                if name in _unary_table:
                    if len(e.args) != 1:
                        raise FPyParserError(loc, 'FPy unary operator expects one argument', e)
                    arg = self._parse_expr(e.args[0])
                    return UnaryOp(_unary_table[name], arg, loc)
                elif name in _binary_table:
                    if len(e.args) != 2:
                        raise FPyParserError(loc, 'FPy binary operator expects two arguments', e)
                    lhs = self._parse_expr(e.args[0])
                    rhs = self._parse_expr(e.args[1])
                    return BinaryOp(_binary_table[name], lhs, rhs, loc)
                elif name in _ternary_table:
                    if len(e.args) != 3:
                        raise FPyParserError(loc, 'FPy ternary operator expects three arguments', e)
                    arg0 = self._parse_expr(e.args[0])
                    arg1 = self._parse_expr(e.args[1])
                    arg2 = self._parse_expr(e.args[2])
                    return TernaryOp(_ternary_table[name], arg0, arg1, arg2, loc)
                elif name == 'or':
                    args = [self._parse_expr(arg) for arg in e.args]
                    return NaryOp(NaryOpKind.OR, args, loc)
                elif name == 'and':
                    args = [self._parse_expr(arg) for arg in e.args]
                    return NaryOp(NaryOpKind.AND, args, loc)
                else:
                    return Call(name, [self._parse_expr(arg) for arg in e.args], loc)
            case ast.Tuple():
                return TupleExpr([self._parse_expr(e) for e in e.elts], loc)
            case ast.List():
                return TupleExpr([self._parse_expr(e) for e in e.elts], loc)
            case ast.ListComp():
                if len(e.generators) != 1:
                    raise FPyParserError(loc, 'FPy only supports a single generator', e)
                elt = self._parse_expr(e.elt)
                var, iterable = self._parse_comprehension(e.generators[0], loc)
                return CompExpr(elt, var, iterable, loc)
            case ast.IfExp():
                cond = self._parse_expr(e.test)
                ift = self._parse_expr(e.body)
                iff = self._parse_expr(e.orelse)
                return IfExpr(cond, ift, iff, loc)
            case _:
                raise NotImplementedError('expression is unsupported in FPy', e)

    def _parse_target(self, target: ast.expr, st: ast.stmt):
        loc = self._parse_location(target)
        match target:
            case ast.Name():
                return target.id
            case ast.Tuple():
                return TupleBinding([self._parse_target(elt, st) for elt in target.elts], loc)
            case _:
                raise FPyParserError(loc, 'FPy expects an identifier', target, st)
 
    def _parse_comprehension(self, gen: ast.comprehension, loc: Location):
        if gen.is_async:
            raise FPyParserError(loc, 'FPy does not support async comprehensions', gen)
        if gen.ifs != []:
            raise FPyParserError(loc, 'FPy does not support if conditions in comprehensions', gen)
        match gen.target:
            case ast.Name():
                return gen.target.id, self._parse_expr(gen.iter)
            case _:
                raise FPyParserError(loc, 'FPy expects an identifier', gen.target, gen)

    def _parse_statement(self, stmt: ast.stmt) -> Stmt:
        """Parse a Python statement."""
        loc = self._parse_location(stmt)
        match stmt:
            case ast.AugAssign():
                if not isinstance(stmt.target, ast.Name):
                    raise FPyParserError(loc, 'Unsupported target in FPy', stmt)
                name = stmt.target.id
                value = self._parse_expr(stmt.value)
                match stmt.op:
                    case ast.Add():
                        return VarAssign(name, BinaryOp(BinaryOpKind.ADD, Var(name, loc), value, loc), None, loc)
                    case ast.Sub():
                        return VarAssign(name, BinaryOp(BinaryOpKind.SUB, Var(name, loc), value, loc), None, loc)
                    case ast.Mult():
                        return VarAssign(name, BinaryOp(BinaryOpKind.MUL, Var(name, loc), value, loc), None, loc)
                    case ast.Div():
                        return VarAssign(name, BinaryOp(BinaryOpKind.DIV, Var(name, loc), value, loc), None, loc)
                    case _:
                        raise FPyParserError(loc, 'Unsupported operator-assignment in FPy', stmt)
            case ast.AnnAssign():
                if not isinstance(stmt.target, ast.Name):
                    raise FPyParserError(loc, 'Unsupported target in FPy', stmt)
                if stmt.annotation is None:
                    raise FPyParserError(loc, 'FPy requires a type annotation', stmt)
                if stmt.value is None:
                    raise FPyParserError(loc, 'FPy requires a value', stmt)
                name = stmt.target.id
                ty = self._parse_type_annotation(stmt.annotation)
                value = self._parse_expr(stmt.value)
                return VarAssign(name, value, ty, loc)
            case ast.Assign():
                if len(stmt.targets) != 1:
                    raise FPyParserError(loc, 'FPy only supports single assignment', stmt)
                binding = self._parse_target(stmt.targets[0], stmt)
                value = self._parse_expr(stmt.value)
                if isinstance(binding, TupleBinding):
                    return TupleAssign(binding, value, loc)
                elif isinstance(binding, str):
                    return VarAssign(binding, value, None, loc)
                else:
                    raise FPyParserError(loc, 'Unexpected binding type', stmt)
            case ast.If():
                cond = self._parse_expr(stmt.test)
                ift = self._parse_statements(stmt.body)
                if stmt.orelse == []:
                    return IfStmt(cond, ift, None, loc)
                else:
                    iff = self._parse_statements(stmt.orelse)
                    return IfStmt(cond, ift, iff, loc)
            case ast.While():
                if stmt.orelse != []:
                    raise FPyParserError(loc, 'FPy does not support else clause in while statement', stmt)
                cond = self._parse_expr(stmt.test)
                block = self._parse_statements(stmt.body)
                return WhileStmt(cond, block, loc)
            case ast.For():
                if stmt.orelse != []:
                    raise FPyParserError(loc, 'FPy does not support else clause in for statement', stmt)
                if not isinstance(stmt.target, ast.Name):
                    raise FPyParserError(loc, 'FPy expects an identifier', stmt)
                var = stmt.target.id
                iterable = self._parse_expr(stmt.iter)
                block = self._parse_statements(stmt.body)
                return ForStmt(var, iterable, block, loc)
            case ast.Return():
                if stmt.value is None:
                    raise FPyParserError(loc, 'Return statement must have value', stmt)
                e = self._parse_expr(stmt.value)
                return Return(e, loc)
            case _:
                raise NotImplementedError('statement is unsupported in FPy', stmt)

    def _parse_statements(self, stmts: list[ast.stmt]):
        """Parse a list of Python statements."""
        return Block([self._parse_statement(s) for s in stmts])

    def _parse_function(self, f: ast.FunctionDef) -> Function:
        """Parse a Python function definition."""
        loc = self._parse_location(f)
        pos_args = f.args.posonlyargs + f.args.args
        if f.args.vararg:
            raise FPyParserError(loc, 'FPy does not support variary arguments', f, f.args.vararg)
        if f.args.kwarg:
            raise FPyParserError(loc, 'FPy does not support keyword arguments', f, f.args.kwarg)

        args: list[Argument] = []
        for arg in pos_args:
            if arg.annotation is None:
                raise FPyParserError(loc, 'FPy requires a type annotation', arg, f)
            name = '_' if arg.arg is None else arg.arg
            ty = self._parse_type_annotation(arg.annotation)
            args.append(Argument(name, ty, loc))

        block = self._parse_statements(f.body)
        return Function(f.name, args, block, loc)
