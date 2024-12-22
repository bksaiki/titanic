"""Analysis or transformation passes on the FPy IR."""

from .define_use import DefineUse
from .for_bundling import ForBundling
from .simplify_if import SimplifyIf
from .verify import VerifyIR
from .while_bundling import WhileBundling
