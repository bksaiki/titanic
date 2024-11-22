from .fpcore import fpy_to_fpcore
from .fpyast import Function
from .parser import fpcore
from .typing import *

### Simple tests

@fpcore
def test_simple1():
    return 0

@fpcore(
    name='Test annotation',
    spec='0.0',
    strict=True
)
def test_simple2():
    return 0

@fpcore(
    name='Test decnum (1/1)',
    spec='0.0',
    strict=True
)
def test_decnum():
    return 0.0

@fpcore(
    name='Test digits (1/4)',
    spec='0.0',
    strict=True
)
def test_digits1():
    return digits(0, 0, 2)

@fpcore(
    name='Test digits (2/4)',
    spec='1.0',
    strict=True
)
def test_digits2():
    return digits(1, 0, 2)

@fpcore(
    name='Test digits (3/4)',
    spec='-2.0',
    strict=True
)
def test_digits3():
    return digits(-1, 0, 2)

@fpcore(
    name='Test digits (4/4)',
    spec='1.5',
    strict=True
)
def test_digits4():
    return digits(3, -1, 2)

@fpcore(
    name='Test let (1/2)',
    spec='1.0',
    strict=True
)
def test_let1():
    a = 1.0
    return a

@fpcore(
    name='Test let (2/2)',
    spec='2.0',
    strict=True
)
def test_let2():
    a = 1.0
    b = 1.0
    return a + b

### Examples

@fpcore(
    name='NMSE example 3.1',
    cite=['hamming-1987', 'herbie-2015'],
    fpbench_domain='textbook',
    # pre=lambda x: x >= 0
    strict=True
)
def nmse3_1(x: Real) -> Real:
    return sqrt(x + 1) - sqrt(x)

# TODO: precondition
@fpcore(
    name='Daisy example instantaneousCurrent',
    cite=['daisy-2018'],
    strict=True
)
def instCurrent(
    t : Real,
    resistance : Real,
    frequency : Real,
    inductance : Real,
    maxVoltage : Real
):
    pi = 3.14159265359
    impedance_re = resistance
    impedance_im = 2 * pi * frequency * inductance
    denom = impedance_re ** 2 + impedance_im ** 2
    current_re = (maxVoltage - impedance_re) / denom
    current_im = (maxVoltage - impedance_im) / denom
    maxCurrent = sqrt(current_re ** 2 + current_im ** 2)
    theta = atan(current_im / current_re)
    return maxCurrent * cos(2 * pi * frequency * t + theta)

### Compile loop

cores: list[Function] = [
    test_simple1,
    test_simple2,
    test_decnum,
    test_digits1,
    test_digits2,
    test_digits3,
    test_digits4,
    test_let1,
    test_let2,
    nmse3_1,
    instCurrent
]

for core in cores:
    fpc = fpy_to_fpcore(core)
    print(fpc.sexp)
