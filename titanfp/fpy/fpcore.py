"""Compilation from FPy to FPCore"""

from typing import Callable, Type

from ..fpbench import fpcast as fpc
from . import fpcore_ops as ops
from .fpyast import *
from .gensym import Gensym

def _compile_argument(arg: Argument):
    match arg.ty:
        case RealType():
            return arg.name, None, None
        case _:
            raise NotImplementedError(arg)

def _compile_compareop(op: CompareOp):
    match op:
        case CompareOp.LT:
            return fpc.LT
        case CompareOp.LE:
            return fpc.LEQ
        case CompareOp.GE:
            return fpc.GEQ
        case CompareOp.GT:
            return fpc.GT
        case CompareOp.EQ:
            return fpc.EQ
        case CompareOp.NE:
            return fpc.NEQ
        case _:
            raise NotImplementedError(op)

def _compile_compare(e: Compare):
    assert e.ops != [], "should not be empty"
    match e.ops:
        case [op]:
            # 2-argument case: just compile
            cls = _compile_compareop(op)
            return cls(_compile_expr(e.children[0]), _compile_expr(e.children[1]))
        case [op, *ops]:
            # N-argument case:
            # TODO: want to evaluate each argument only once;
            #       may need to let-bind in case any argument is
            #       used multiple times
            args = [_compile_expr(arg) for arg in e.children]

            # gensym = Gensym()
            # for arg in args:
            #     if isinstance(arg, fpc.Var):
            #         gensym.reserve(arg.name)

            curr_group = (op, [args[0], args[1]])
            groups: list[tuple[CompareOp, list[fpc.Expr]]] = [curr_group]
            # let_binds: dict[str, fpc.Expr] = {}

            for op, lhs, rhs in zip(ops, args[1:], args[2:]):
                if op == curr_group[0] or isinstance(lhs, fpc.ValueExpr):
                    # same op => append
                    # different op (terminal) => append
                    curr_group[1].append(lhs)
                else:
                    # different op (non-terminal) => new group
                    new_group = (op, [lhs, rhs])
                    groups.append(new_group)
                    curr_group = new_group
                
            if len(groups) == 1:
                op, args = groups[0]
                cls = _compile_compareop(op)
                return cls(*args)
            else:
                return fpc.And(*[_compile_compareop(op)(*args) for op, args in groups])


def _compile_expr(e: Expr) -> fpc.Expr:
    match e:
        case IfExpr(cond=cond, ift=ift, iff=iff):
            return fpc.If(_compile_expr(cond), _compile_expr(ift), _compile_expr(iff))
        case Compare():
            return _compile_compare(e)
        case ArrayExpr(children=children):
            return fpc.Array(*map(_compile_expr, children))
        case UnknownCall(name=name, children=children):
            return fpc.UnknownOperator(name=name, *map(_compile_expr, children))
        case NaryExpr(name=name, children=children):
            if type(e) not in ops.known_table:
                raise NotImplementedError('no compilation method for', e)        
            cls = ops.known_table[type(e)]
            return cls(*map(_compile_expr, children))
        case Var(name=name):
            return fpc.Var(name)
        case Num(val=val):
            if isinstance(val, int):
                return fpc.Integer(val)
            elif isinstance(val, float):
                return fpc.Decnum(str(val))
            else:
                raise NotImplementedError(e)
        case Digits(m=m, e=exp, b=base):
            return fpc.Digits(m, exp, base)
        case _:
            raise NotImplementedError(e)
        
def _make_tuple_binding(tuple_id: str, binding: TupleBinding, pos: list[int]):
    tuple_binds: list[tuple[str, fpc.Expr]] = []
    for i, bind in enumerate(binding.bindings):
        match bind:
            case TupleBinding():
                tuple_binds += _make_tuple_binding(tuple_id, bind, [i, *pos])
            case VarBinding(name=name):
                idxs = [fpc.Integer(idx) for idx in [i, *pos]]
                tuple_binds.append((name, fpc.Ref(fpc.Var(tuple_id), *idxs)))
            case _:
                raise NotImplementedError('unexpected binding', bind)
    return tuple_binds

def _compile_block(block: Block):
   match block.stmts:
        case [stmt]:
            return _compile_statement(stmt)
        case [stmt, *stmts]:
            match stmt:
                case Assign(var=var, val=val):
                    return fpc.Let([(var.name, _compile_expr(val))], _compile_block(Block(stmts)))
                case TupleAssign(binding=binding, val=val):
                    tuple_id = 't0' # TODO: needs to be a unique identifier
                    tuple_bind = (tuple_id, _compile_expr(val))
                    destruct_bindings = _make_tuple_binding(tuple_id, binding, [])
                    return fpc.Let([tuple_bind] + destruct_bindings, _compile_block(Block(stmts)))
                case _:
                    raise NotImplementedError('unexpected statement', stmt)
        case _:
           raise NotImplementedError('unexpected block', block.stmts)

def _compile_statement(stmt: Stmt) -> fpc.Expr:
    match stmt:
        case Block():
            return _compile_block(stmt)
        case Return(e=expr):
            return _compile_expr(expr)
        case _:
            raise NotImplementedError(stmt)

def _compile_func(func: Function):
    args = list(map(_compile_argument, func.args))
    # TODO: parse data
    props = func.ctx.props
    e = _compile_statement(func.body)

    return fpc.FPCore(
        inputs=args,
        e=e,
        props=props,
        ident=func.ident,
        name=func.name,
        pre=func.pre
    )

def fpy_to_fpcore(func: Function):
    if not isinstance(func, Function):
        raise TypeError(f'expected Function: {func}')
    return _compile_func(func)
