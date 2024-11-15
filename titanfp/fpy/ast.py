"""AST nodes in the FPY language"""

from abc import ABC
from dataclasses import dataclass

from ..titanic.digital import Digital

class Ast(ABC):
    """Abstract base class for FPY AST nodes."""

    def __add__(self, other):
        return Add(self, other)
    
    def __sub__(self, other):
        return Sub(self, other)
    
    def __mul__(self, other):
        return Mul(self, other)
    
    def __div__(self, other):
        return Div(self, other)

@dataclass
class Real(Ast):
    """FPY node: numerical constant."""
    val: Digital

@dataclass
class Add(Ast):
    lhs: Ast
    rhs: Ast

@dataclass
class Sub(Ast):
    lhs: Ast
    rhs: Ast

@dataclass
class Mul(Ast):
    lhs: Ast
    rhs: Ast

@dataclass
class Div(Ast):
    lhs: Ast
    rhs: Ast
