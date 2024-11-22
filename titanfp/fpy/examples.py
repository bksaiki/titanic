from .parser import fpcore
from .typing import *

### Sanity testing

@fpcore
def simple1():
    return 0

@fpcore(
    name='Test annotation',
    spec='0.0',
    strict=True
)
def simple2():
    return 0

@fpcore(
    name='Test decnum (1/1)',
    spec='0.0',
    strict=True
)
def decnum():
    return 0.0

@fpcore(
    name='Test digits (1/4)',
    spec='0.0',
    strict=True
)
def digits1():
    return digits(0, 0, 2)

@fpcore(
    name='Test digits (2/4)',
    spec='1.0',
    strict=True
)
def digits2():
    return digits(1, 0, 2)

@fpcore(
    name='Test digits (3/4)',
    spec='-2.0',
    strict=True
)
def digits3():
    return digits(-1, 0, 2)

@fpcore(
    name='Test digits (4/4)',
    spec='1.5',
    strict=True
)
def digits4():
    return digits(3, -1, 2)

@fpcore(
    name='Test let (1/2)',
    spec='1.0',
    strict=True
)
def let1():
    a = 1.0
    return a

@fpcore(
    name='Test let (2/2)',
    spec='2.0',
    strict=True
)
def let2():
    a = 1.0
    b = 1.0
    return a + b

@fpcore(
    name='NMSE example 3.1',
    cite=['hamming-1987', 'herbie-2015'],
    fpbench_domain='textbook',
    # pre=lambda x: x >= 0
)
def nmse3_1(x: Real) -> Real:
    return sqrt(x + 1) - sqrt(x)
