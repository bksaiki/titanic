"""Defines the `@fpcore` decorator for FPy functions."""

import inspect

from inspect import Parameter
from typing import Any, Callable, Optional

from .ast import Bool, Real, FPCore
from .utils import raise_type_error

_ParameterKind = type(Parameter.KEYWORD_ONLY)
_ParamDict = dict[str, tuple[_ParameterKind, Optional[type]]]
_TypeDict = dict[str, type]

def _extract_signature(func: Callable[..., Any]):
    sig = inspect.signature(func)

    args: _ParamDict = {}
    for name, param in sig.parameters.items():
        if param.annotation == Parameter.empty:
            args[name] = (param.kind, None)
        else:
            args[name] = (param.kind, param.annotation)

    return args

def _is_value_type(ty: type):
    return ty == Bool or ty == Real

def _signature_is_compliant(args: _ParamDict):
    is_compliant = True
    for kind, ty in args.values():
        match kind:
            case Parameter.POSITIONAL_ONLY:
                if ty is None or not _is_value_type(ty):
                    is_compliant = False
            case Parameter.POSITIONAL_OR_KEYWORD:
                if ty is None or not _is_value_type(ty):
                    is_compliant = False
            case _: # should any of these be handled?
                is_compliant = True

    return is_compliant

def _signature_as_typedict(args: _ParamDict):
    tydict: _TypeDict = {}
    for name, (_, ty) in args.items():
        assert ty is not None
        tydict[name] = ty
    return tydict

def fpcore(func: Callable[..., Any]):
    """
    Constructs a callable (and inspectable) FPY function from
    an arbitrary Python function. If possible, the argument
    is analyzed for type information. The result is either
    the function as-is or a callable `FPCore` instance that
    has additional funcitonality.
    """
    if not callable(func):
        raise_type_error(function, func)

    sig = _extract_signature(func)
    if not _signature_is_compliant(sig):
        raise RuntimeError(f'FPCore {func} signature not compliant', sig)
    
    arg_tys = _signature_as_typedict(sig)
    source = '\n'.join(inspect.getsource(func).splitlines()[1:])
    return FPCore(func, arg_tys, source)
