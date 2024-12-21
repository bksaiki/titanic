"""
Decorators for the FPy language.
"""

import inspect

from .codegen import IRCodegen
from .definition import DefinitionAnalysis
from .fpyast import Function
from .live_vars import LiveVarAnalysis
from .parser import Parser

from ..passes import VerifyIR, DefineUse

def fpcore(*args, **kwargs):
    """
    Decorator to parse a Python function into FPy.

    Constructs an FPy `Function` from a Python function.
    FPy is a stricter subset of Python, so this decorator will reject
    any function that is not valid in FPy.
    """

    def decorator(func):
        if not callable(func):
            raise TypeError('fpcore() requires a callable object')

        # read the original source of the function
        sourcename = inspect.getabsfile(func)
        lines, start_line = inspect.getsourcelines(func)
        source = ''.join(lines)
        
        # parse the source as an FPy function
        parser = Parser(sourcename, source, start_line)
        ast = parser.parse()
        assert isinstance(ast, Function), "must be a function"

        # analyze and lower to the IR
        DefinitionAnalysis().analyze(ast)
        LiveVarAnalysis().analyze(ast)
        ir = IRCodegen.lower(ast)
        VerifyIR.check(ir)
        return ir

    # handle any arguments to the decorator
    match args:
        case []:
            return decorator
        case [f]:
            return decorator(f)
        case _:
            raise TypeError('fpcore() takes 0 or 1 argument')
