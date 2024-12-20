from .fpcore import fpy_to_fpcore
from .fpyast import Function
from .frontend.decorator import fpcore
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
    return (1.0, 2.0, 3.0)

@fpcore
def test_array2():
    return 1.0, 2.0, 3.0

@fpcore
def test_array3():
    x, y = 1.0, 2.0
    return x + y

@fpcore
def test_array4():
    x, y = (1.0, 2.0), (3.0, 4.0)
    x0, x1 = x
    y0, y1 = y
    return x0 * y0 + x1 * y1

@fpcore(name='Test if statement (1/4)')
def test_if1():
    t = 0
    if 0 < 1:
        t = 1
    return t

@fpcore(name='Test if statement (2/4)')
def test_if2():
    if 0 < 1:
        t = 1
    else:
        t = 0
    return t

@fpcore(name='Test if statement (3/4)')
def test_if3():
    if 0 < 1:
        if 1 < 2:
            t = 0
        else:
            t = 1
    else:
        if 2 > 1:
            t = 2
        else:
            t = 3
    return t

@fpcore(name='Test if statement (4/4)')
def test_if4():
    if 0 < 1:
        t = 0
    elif 1 < 2:
        t = 1
    else:
        t = 2
    return t

@fpcore
def test_while1():
    x = 0
    while x < 1:
        x = 1
    return x

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

# TODO: vectors should be tensors
@fpcore(
    name='Level-of-detail (LOD) algorithm, anisotropic case',
    cite=['DirectX 11.3 specification, Microsoft-2015'],
    strict=True
)
def lod_anisotropic(
    dx_u : Real,
    dx_v : Real,
    dy_u : Real,
    dy_v : Real,
    max_aniso : Real,
):
    dx2 = dx_u ** 2 + dx_v ** 2
    dy2 = dy_u ** 2 + dy_v ** 2
    det = fabs(dx_u * dy_v - dx_v * dy_u)
    x_major = dx2 > dy2
    major2 = dx2 if x_major else dy2
    major = sqrt(major2)
    norm_major = 1.0 / major

    aniso_dir_u = (dx_u if x_major else dy_u) * norm_major
    aniso_dir_v = (dx_v if x_major else dy_v) * norm_major
    aniso_ratio = major2 / det

    # clamp anisotropy ratio and compute LOD
    if aniso_ratio > max_aniso:
        aniso_ratio = max_aniso
        minor = major / aniso_ratio
    else:
        minor = det / major
    
    # clamp LOD
    if minor < 1.0:
        aniso_ratio = fmax(1.0, aniso_ratio * minor)

    lod = log2(minor)

    return lod, aniso_ratio, aniso_dir_u, aniso_dir_v


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
    test_array4,
    test_if1,
    test_if2,
    test_if3,
    test_if4,
    # Examples
    nmse3_1,
    instCurrent,
    azimuth,
    lod_anisotropic
]

for core in cores:
    print(core)
    # fpc = fpy_to_fpcore(core)
    # print(fpc.sexp)
