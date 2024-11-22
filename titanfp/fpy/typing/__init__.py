"""
Typing hints for FPy programs.

FPy programs are re-parsed and analyzing with
the `@fpcore` decorator provided by `titanfp.fpy`.
Thus, functions in FPy programs don't actually exist in the runtime.

To help static analyzers like Pylance and mypy,
this module provides a stub file with typing hints
for FPy functions.

Use the hints:

>>  > from titanfp.fpy.typing import *

The names in this module must be imported directly into
the importing namespace.
"""

from abc import ABC

class Real(ABC):
    """FPy typing hint for real values."""
    pass

