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

@fpcore(
  name='Test if expression (1/6)',
  spec='1.0'
)
def test_ife1():
  return 1.0 if 1.0 > 0.0 else 0.0

@fpcore(name='Test if expression (2/6)')
def test_ife2():
  return 1.0 if 0.0 < 1.0 < 2.0 else 0.0

@fpcore(name='Test if expression (3/6)')
def test_ife3():
  x = 1.0
  y = 2.0
  z = 3.0
  t = 4.0
  return 1.0 if (x + 1.0) < (y < 2.0) < (z + 3.0) < (t + 4.0) else 0.0

@fpcore(name='Test if expression (4/6)')
def test_ife4():
  x = 1.0
  y = 2.0
  z = 3.0
  t = 4.0
  return 1.0 if (x + 1.0) < (y + 2.0) <= (z + 3.0) < (t + 4.0) else 0.0

@fpcore
def test_array1():
    return (1.0, 2.0)

@fpcore
def test_array2():
    return 1.0, 2.0, 3.0

@fpcore
def test_array3():
    x, y = 1.0, 2.0
    return x + y

@fpcore(name='Test if statement')
def test_if():
    if 0.0 < 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

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

@fpcore(
    name='azimuth',
    cite=['solovyev-2015'],
    strict=True
)
def azimuth(lat1: Real, lat2: Real, lon1: Real, lon2: Real):
    dLon = lon2 - lon1
    s_lat1 = sin(lat1)
    c_lat1 = cos(lat1)
    s_lat2 = sin(lat2)
    c_lat2 = cos(lat2)
    s_dLon = sin(dLon)
    c_dLon = cos(dLon)
    return atan((c_lat2 * s_dLon) / ((c_lat1 * s_lat2) - (s_lat1 * c_lat2 * c_dLon)))

### Compile loop

cores: list[Function] = [
    # Tests
    test_simple1,
    test_simple2,
    test_decnum,
    test_digits1,
    test_digits2,
    test_digits3,
    test_digits4,
    test_let1,
    test_let2,
    test_ife1,
    test_ife2,
    test_ife3,
    test_ife4,
    test_array1,
    test_array2,
    test_array3,
    test_if,
    # Examples
    nmse3_1,
    instCurrent,
    azimuth
]

for core in cores:
    fpc = fpy_to_fpcore(core)
    print(fpc.sexp)

# (FPCore newton-raphson (x0 tolerance)
#  (while (> (fabs (- x1 x0)) tolerance) 
#   ([x0 x0 x1]
#    [x1 (- x0 (/ (f x0) (fprime x0)))
#        (- x1 (/ (f x1) (fprime x1)))])
#   x1)
# )

# def newton(x0 : Real, eps : Real):
#     x1 = x0 - (f(x0) / fprime(x0))
#     while eps < fabs(x1 - x0):
#         x0 = x1
#         x1 = x1 - (f(x1) / fprime(x1))
#     return x1


# def newton2(x0, eps):
#     x1 = x0 - (f(x0) / fprime(x0))
#     with While(eps < fabs(x1 - x0)) as result:
#         x0 = x1
#         x1 = x1 - (f(x1) / fprime(x1))
#     return x1




# (FPCore newton2 (x0 tolerance)
#   (let ([x1 ...])
#    (let ([env
         #     (while <cond> ([env <update env>])
         #       (array x0 x1))])
         # (let ([x0 (ref env 0)]
        #        [x1 (ref env 1)])
        #    x1))))
