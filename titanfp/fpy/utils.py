

from typing import Any

class FPySyntaxError(Exception):
    """Syntax error for FPy programs."""
    pass

def raise_type_error(ty: type, v: Any):
    expect_name = ty.__name__
    actual_name = type(v).__name__
    raise TypeError(f'expected value of type {expect_name}, got {v} (type: {actual_name})')
