"""
FPy IR types
"""

from typing import Self

class IRType:
    """FPy IR: IR type"""

    def __repr__(self):
        name = self.__class__.__name__
        items = ', '.join(f'{k}={repr(v)}' for k, v in self.__dict__.items())
        return f'{name}({items})'

class AnyType(IRType):
    """FPy IR: any type"""

class RealType(IRType):
    """FPy IR: real type"""

class BoolType(IRType):
    """FPy IR: boolean type"""

class TensorType(IRType):
    """FPy IR: tensor type"""
    elts: list[IRType | Self]

    def __init__(self, elts: list[IRType | Self]):
        self.elts = elts
