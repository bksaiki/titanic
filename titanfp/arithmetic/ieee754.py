"""Emulated IEEE 754 floating-point arithmetic.
"""

from inspect import getargspec
import multiprocessing as mp
import gmpy2 as gmp

from ..titanic import gmpmath
from ..titanic import digital

from ..titanic.integral import bitmask
from ..titanic.ops import RM, OP
from .evalctx import EvalCtx, IEEECtx
from . import mpnum
from . import interpreter


used_ctxs = {}
def ieee_ctx(es, nbits, rm=RM.RNE):
    try:
        return used_ctxs[(es, nbits, rm)]
    except KeyError:
        ctx = IEEECtx(es=es, nbits=nbits, rm=rm)
        used_ctxs[(es, nbits, rm)] = ctx
        return ctx

class IEEEFlags(object):
    """Status flags defined by IEEE-754"""
    invalid: bool = False
    divzero: bool = False
    overflow: bool = False
    underflow: bool = False
    inexact: bool = False

    def __init__(self, invalid, divzero, overflow, underflow, inexact):
        self.invalid = invalid
        self.divzero = divzero
        self.overflow = overflow
        self.underflow = underflow
        self.inexact = inexact

    def __repr__(self):
        return '{}(invalid={}, divzero={}, overflow={}, underflow={}, inexact={})'.format(
            type(self).__name__, self.invalid, self.divzero, self.overflow, self.underflow, self.inexact
        )


class Float(mpnum.MPNum):
    """IEEE-754 floating point type"""

    # rounding context at construction
    _ctx : IEEECtx = ieee_ctx(11, 64)

    # IEEE-754 required flags
    _invalid: bool = False      # real result was undefined => NaN
    _divzero: bool = False      # pole with well-defined limit => +/-Inf
    _overflow: bool = False     # magnitude was too large 
    _underflow: bool = False    # magnitude was too small

    # (already defined in Digital)
    # _inexact: bool = False

    @property
    def ctx(self):
        """The rounding context used to compute this value.
        If a computation takes place between two values, then
        it will either use a provided context (which will be recorded
        on the result) or the more precise of the parent contexts
        if none is provided.
        """
        return self._ctx

    def is_identical_to(self, other):
        if isinstance(other, type(self)):
            return super().is_identical_to(other) and self.ctx.es == other.ctx.es and self.ctx.nbits == other.ctx.nbits
        else:
            return super().is_identical_to(other)

    def __init__(self, x=None, ctx=None, **kwargs):
        if ctx is None:
            ctx = type(self)._ctx

        if x is None or isinstance(x, digital.Digital):
            super().__init__(x=x, **kwargs)
            self._invalid = self._divzero
            self._divzero = False
            self._overflow = self._isinf and self._inexact
            self._underflow = type(self)._check_underflow(self, ctx) and self._inexact
        else:
            if kwargs:
                raise ValueError('cannot specify additional values {}'.format(repr(kwargs)))
            unrounded = gmpmath.mpfr_to_digital(gmpmath.mpfr(x, ctx.p))
            rounded = self._round_to_context(unrounded, ctx=ctx, strict=True)
            super().__init__(x=rounded)
            self._invalid = rounded._invalid
            self._divzero = rounded._divzero
            self._overflow = rounded._overflow
            self._underflow = rounded._underflow
            

        self._ctx = ieee_ctx(ctx.es, ctx.nbits, rm=ctx.rm)

    def __repr__(self):
        return '{}(negative={}, c={}, exp={}, inexact={}, rc={}, isinf={}, isnan={}, ctx={})'.format(
            type(self).__name__, repr(self._negative), repr(self._c), repr(self._exp),
            repr(self._inexact), repr(self._rc), repr(self._isinf), repr(self._isnan), repr(self._ctx)
        )

    def __str__(self):
        return str(gmpmath.digital_to_mpfr(self))

    def __float__(self):
        return float(gmpmath.digital_to_mpfr(self))

    def __int__(self):
        return int(gmpmath.digital_to_mpfr(self))

    @classmethod
    def _select_context(cls, *args, ctx=None):
        if ctx is not None:
            return ieee_ctx(ctx.es, ctx.nbits, rm=ctx.rm)
        else:
            es = max((f.ctx.es for f in args if isinstance(f, cls)))
            p = max((f.ctx.p for f in args if isinstance(f, cls)))
            return ieee_ctx(es, es + p)

    @classmethod
    def _round_to_context(cls, unrounded, ctx=None, strict=False):
        if ctx is None:
            if isinstance(unrounded, cls):
                ctx = unrounded.ctx
            else:
                raise ValueError('no context specified to round {}'.format(repr(unrounded)))

        # NaN
        if unrounded.isnan:
            result = cls(unrounded, ctx=ctx)
            result._invalid = True
            result._divzero = False
            result._overflow = False
            result._underflow = False
            return result

        # Inf
        if unrounded.isinf:
            # TODO: how to set _divzero ???
            result = cls(unrounded, ctx=ctx)
            result._invalid = False
            result._divzero = False
            result._overflow = unrounded._inexact
            result._underflow = False
            return result

        # check for overflow
        overflow, round_away = cls._check_overflow(unrounded, ctx)
        if overflow:
            if round_away:
                # round to infinity
                result = cls(negative=unrounded.negative, isinf=True, inexact=True, ctx=ctx)
                result._invalid = False
                result._divzero = False
                result._overflow = True
                result._underflow = False
                return result
            else:
                # clamp to maxval
                result = cls(negative=unrounded.negative, x=ctx.maxnorm, inexact=True, ctx=ctx)
                result._invalid = False
                result._divzero = False
                result._overflow = True
                result._underflow = False
                return result

        # check for underflow
        underflow = cls._check_underflow(unrounded, ctx)

        # normal
        result = cls(unrounded.round_new(max_p=ctx.p, min_n=ctx.n, rm=ctx.rm, strict=strict), ctx=ctx)
        result._invalid = False
        result._divzero = False
        result._overflow = overflow
        result._underflow = underflow and result._inexact
        return result

    # overflow and underflow

    @classmethod
    def _check_overflow(cls, unrounded: digital.Digital, ctx: IEEECtx):
        magnitude = digital.Digital(x=unrounded, negative=False)
        cmp_maxnorm = magnitude.compareto_exact(ctx.maxnorm)
        if cmp_maxnorm is None or cmp_maxnorm <= 0:
            return False, False

        nearest, mode = cls._rounding_modes[(unrounded._negative, ctx.rm)]
        if nearest:
            overflow = magnitude.compareto_exact(ctx.maxbound) >= 0
            round_away = True
        elif mode == digital.RoundingMode.TOWARD_ZERO:
            overflow = magnitude.compareto_exact(ctx.infval) >= 0
            round_away = False
        else:   #    digital.RoundingMode.AWAY_ZERO
            overflow = True
            round_away = True

        return overflow, round_away

    @classmethod
    def _check_underflow(cls, unrounded: digital.Digital, ctx: IEEECtx):
        magnitude = digital.Digital(x=unrounded, negative=False)
        cmp_minnorm = magnitude.compareto_exact(ctx.minnorm)
        if cmp_minnorm is None or cmp_minnorm >= 0:
            return False

        nearest, mode = cls._rounding_modes[(unrounded._negative, ctx.rm)]
        if nearest:
            return magnitude.compareto_exact(ctx.subbound) < 0
        elif mode == digital.RoundingMode.TOWARD_ZERO:
            return True
        else:   #    digital.RoundingMode.AWAY_ZERO
            return magnitude.compareto_exact(ctx.tinyval) <= 0

    # status flags

    def set_divzero(self, *args):
        self._divzero = self._isinf and not self._inexact and all(map(lambda x : x.is_finite_real(), args))

    def get_flags(self):
        return IEEEFlags(self._invalid, self._divzero, self._overflow, self._underflow, self._inexact)

    # most operations come from mpnum

    def isnormal(self):
        return not (
            self.is_zero()
            or self.isinf
            or self.isnan
            or self.e < self.ctx.emin
        )

    def compute(self, fn, *args, ctx=ctx):
        try:
            result = fn(*args, ctx=ctx)
            result.set_divzero(self, *args)
        except gmpmath.OverflowResultError as err:
            ctx = type(self)._select_context(*args, ctx=ctx)
            result = Float(isinf=True, inexact=True, negative=err.sign, ctx=ctx)
        except gmpmath.UnderflowResultError as err:
            ctx = type(self)._select_context(*args, ctx=ctx)
            result = Float(c=0, exp=0, inexact=True, negative=err.sign, ctx=ctx)
        return result

    def add(self, other, ctx=None):
        result = self.compute(super().add, other, ctx=ctx)
        if not result.inexact and result.is_finite_real() and result.is_zero() and self.negative != other.negative:
            # IEEE-754 standard says x + -x = -0 for RTN and +0 otherwise
            ctx = type(self)._select_context(self, other, ctx=ctx)
            result = type(self)(x=result, negative=(ctx.rm == RM.RTN))
        return result

    def sub(self, other, ctx=None):
        result = self.compute(super().sub, other, ctx=ctx)
        if not result.inexact and result.is_finite_real() and result.is_zero() and self.negative == other.negative:
            # IEEE-754 standard says x - x = -0 for RTN and +0 otherwise
            ctx = type(self)._select_context(self, other, ctx=ctx)
            result = type(self)(x=result, negative=(ctx.rm == RM.RTN))
        return result

    def mul(self, other, ctx=None):
        return self.compute(super().mul, other, ctx=ctx)

    def div(self, other, ctx=None):
        return self.compute(super().div, other, ctx=ctx)

    def sqrt(self, ctx=None):
        return self.compute(super().sqrt, ctx=ctx)

    def fma(self, other1, other2, ctx=None):
        result = self.compute(super().fma, other1, other2, ctx=ctx)
        mul_sign = self.negative != other1.negative
        if not result.inexact and result.is_finite_real() and result.is_zero() and mul_sign != other2.sign:
            # IEEE-754 standard says x + -x = -0 for RTN and +0 otherwise
            ctx = type(self)._select_context(self, other1, other2, ctx=ctx)
            result = type(self)(x=result, negative=(ctx.rm == RM.RTN))
        return result

    def neg(self, ctx=None):
        return self.compute(super().neg, ctx=ctx)

    def copysign(self, other, ctx=None):
        return self.compute(super().copysign, other, ctx=ctx)

    def fabs(self, ctx=None):
        return self.compute(super().fabs, ctx=ctx)

    def fdim(self, other, ctx=None):
        return self.compute(super().fdim, other, ctx=ctx)

    def fmax(self, other, ctx=None):
        return self.compute(super().fmax, other, ctx=ctx)

    def fmin(self, other, ctx=None):
        return self.compute(super().fmin, other, ctx=ctx)

    def fmod(self, other, ctx=None):
        return self.compute(super().fmod, other, ctx=ctx)

    def remainder(self, other, ctx=None):
        return self.compute(super().remainder, other, ctx=ctx)

    def ceil(self, ctx=None):
        return self.compute(super().ceil, ctx=ctx)

    def floor(self, ctx=None):
        return self.compute(super().floor, ctx=ctx)

    def nearbyint(self, ctx=None):
        return self.compute(super().nearbyint, ctx=ctx)

    def round(self, ctx=None):
        return self.compute(super().round, ctx=ctx)

    def trunc(self, ctx=None):
        return self.compute(super().trunc, ctx=ctx)

    def acos(self, ctx=None):
        return self.compute(super().acos, ctx=ctx)

    def acosh(self, ctx=None):
        return self.compute(super().acosh, ctx=ctx)

    def atan(self, ctx=None):
        return self.compute(super().atan, ctx=ctx)

    def atan2(self, other, ctx=None):
        return self.compute(super().atan2, other, ctx=ctx)

    def atanh(self, ctx=None):
        return self.compute(super().atan2, ctx=ctx)

    def cos(self, ctx=None):
        return self.compute(super().cos, ctx=ctx)

    def cosh(self, ctx=None):
        return self.compute(super().cosh, ctx=ctx)

    def sin(self, ctx=None):
        return self.compute(super().sin, ctx=ctx)

    def sinh(self, ctx=None):
        return self.compute(super().sinh, ctx=ctx)

    def tan(self, ctx=None):
        return self.compute(super().tan, ctx=ctx)

    def tanh(self, ctx=None):
        return self.compute(super().tanh, ctx=ctx)

    def exp_(self, ctx=None):
        return self.compute(super().exp_, ctx=ctx)

    def exp2(self, ctx=None):
        return self.compute(super().exp2, ctx=ctx)

    def expm1(self, ctx=None):
        return self.compute(super().expm1, ctx=ctx)

    def log(self, ctx=None):
        return self.compute(super().log, ctx=ctx)

    def log10(self, ctx=None):
        return self.compute(super().log10, ctx=ctx)

    def log1p(self, ctx=None):
        return self.compute(super().log1p, ctx=ctx)

    def log2(self, ctx=None):
        return self.compute(super().log2, ctx=ctx)

    def cbrt(self, ctx=None):
        return self.compute(super().cbrt, ctx=ctx)

    def hypot(self, other, ctx=None):
        return self.compute(super().hypot, other, ctx=ctx)

    def pow(self, other, ctx=None):
        return self.compute(super().pow, other, ctx=ctx)

    def erf(self, ctx=None):
        return self.compute(super().erf, ctx=ctx)

    def erfc(self, ctx=None):
        return self.compute(super().erfc, ctx=ctx)

    def lgamma(self, ctx=None):
        return self.compute(super().lgamma, ctx=ctx)

    def tgamma(self, ctx=None):
        return self.compute(super().tgamma, ctx=ctx)


class Interpreter(interpreter.StandardInterpreter):
    dtype = Float
    ctype = IEEECtx

    def arg_to_digital(self, x, ctx):
        return self.dtype(x, ctx=ctx)

    def _eval_constant(self, e, ctx):
        try:
            return None, self.constants[e.value]
        except KeyError:
            return None, self.round_to_context(gmpmath.compute_constant(e.value, prec=ctx.p), ctx=ctx)

    # unfortunately, interpreting these values efficiently requries info from the context,
    # so it has to be implemented per interpreter...

    def _eval_integer(self, e, ctx):
        x = digital.Digital(m=e.i, exp=0, inexact=False)
        return None, self.round_to_context(x, ctx=ctx)

    def _eval_rational(self, e, ctx):
        p = digital.Digital(m=e.p, exp=0, inexact=False)
        q = digital.Digital(m=e.q, exp=0, inexact=False)
        x = gmpmath.compute(OP.div, p, q, prec=ctx.p)
        return None, self.round_to_context(x, ctx=ctx)

    def _eval_digits(self, e, ctx):
        x = gmpmath.compute_digits(e.m, e.e, e.b, prec=ctx.p)
        return None, self.round_to_context(x, ctx=ctx)

    def round_to_context(self, x, ctx):
        """Not actually used?"""
        return self.dtype._round_to_context(x, ctx=ctx, strict=False)

    def _eval_exp(self, e, ctx):
        in0 = self.evaluate(e.children[0], ctx)
        return [in0], in0.exp_(ctx=ctx)


def digital_to_bits(x, ctx=None):
    if ctx.p < 2 or ctx.es < 2:
        raise ValueError('format with w={}, p={} cannot be represented with IEEE 754 bit pattern'.format(ctx.es, ctx.p))

    if ctx is None:
        rounded = x
        ctx = x.ctx
    else:
        rounded = x._round_to_context(x, ctx=ctx, strict=False)

    pbits = ctx.p - 1

    if rounded.negative:
        S = 1
    else:
        S = 0

    if rounded.isnan:
        # canonical NaN
        return (0 << (ctx.es + pbits)) | (bitmask(ctx.es) << pbits) | (1 << (pbits - 1))
    elif rounded.isinf:
        return (S << (ctx.es + pbits)) | (bitmask(ctx.es) << pbits) # | 0
    elif rounded.is_zero():
        return (S << (ctx.es + pbits)) # | (0 << pbits) | 0

    c = rounded.c
    cbits = rounded.p
    e = rounded.e

    if e < ctx.emin:
        # subnormal
        lz = (ctx.emin - 1) - e
        if lz > pbits or (lz == pbits and cbits > 0):
            raise ValueError('exponent out of range: {}'.format(e))
        elif lz + cbits > pbits:
            raise ValueError('too much precision: given {}, can represent {}'.format(cbits, pbits - lz))
        E = 0
        C = c << (lz - (pbits - cbits))
    elif e <= ctx.emax:
        # normal
        if cbits > ctx.p:
            raise ValueError('too much precision: given {}, can represent {}'.format(cbits, ctx.p))
        elif cbits < ctx.p:
            raise ValueError('too little precision: given {}, can represent {}'.format(cbits, ctx.p))
        E = e + ctx.emax
        C = (c << (ctx.p - cbits)) & bitmask(pbits)
    else:
        # overflow
        raise ValueError('exponent out of range: {}'.format(e))

    return (S << (ctx.es + pbits)) | (E << pbits) | C

def bits_to_digital(i, ctx=ieee_ctx(11, 64)):
    pbits = ctx.p - 1

    S = (i >> (ctx.es + pbits)) & bitmask(1)
    E = (i >> pbits) & bitmask(ctx.es)
    C = i & bitmask(pbits)

    negative = (S == 1)
    e = E - ctx.emax

    if E == 0:
        # subnormal
        c = C
        exp = -ctx.emax - pbits + 1
    elif e <= ctx.emax:
        # normal
        c = C | (1 << pbits)
        exp = e - pbits
    else:
        # nonreal
        if C == 0:
            return Float(ctx=ctx, negative=negative, c=0, exp=0, isinf=True)
        else:
            return Float(ctx=ctx, negative=False, c=0, exp=0, isnan=True)

    # unfortunately any rc / exactness information is lost
    return Float(ctx=ctx, negative=negative, c=c, exp=exp, rounded=False, inexact=False)

def show_bitpattern(x, ctx=None):
    if isinstance(x, int):
        if ctx is None:
            ctx = ieee_ctx(11, 64)
        i = x
    elif isinstance(x, Float):
        if ctx is None:
            ctx = x.ctx
        i = digital_to_bits(x, ctx=ctx)

    S = i >> (ctx.es + ctx.p - 1)
    E = (i >> (ctx.p - 1)) & bitmask(ctx.es)
    C = i & bitmask(ctx.p - 1)
    if E == 0 or E == bitmask(ctx.es):
        hidden = 0
    else:
        hidden = 1

    return ('float{:d}({:d},{:d}): {:01b} {:0'+str(ctx.es)+'b} ({:01b}) {:0'+str(ctx.p-1)+'b}').format(
        ctx.es + ctx.p, ctx.es, ctx.p, S, E, hidden, C,
    )

#
#   Testing
#

fl_ops = [
    Float.add,
    Float.sub,
    Float.mul,
    Float.div,
    Float.neg,
    Float.sqrt,
    Float.fma,
    Float.copysign,
    Float.fabs,
    Float.fdim,
    Float.fmax,
    Float.fmin,
    Float.fmod,
    Float.remainder,
    Float.ceil,
    Float.floor,
    Float.nearbyint,
    Float.round,
    Float.trunc,
    Float.acos,
    Float.acosh,
    Float.asin,
    Float.asinh,
    Float.atan,
    Float.atan2,
    Float.atanh,
    Float.cos,
    Float.cosh,
    Float.sin,
    Float.sinh,
    Float.tan,
    Float.tanh,
    Float.exp_,
    Float.exp2,
    Float.expm1,
    Float.log,
    Float.log10,
    Float.log1p,
    Float.log2,
    Float.cbrt,
    Float.hypot,
    Float.pow,
    Float.erf,
    Float.erfc,
    Float.lgamma,
    Float.tgamma,
]

gmp_rms = {
    RM.RNE: gmp.RoundToNearest,
    RM.RTP: gmp.RoundUp,
    RM.RTN: gmp.RoundDown,
    RM.RTZ: gmp.RoundToZero,
    RM.RAZ: gmp.RoundAwayZero
}

def mpfr_compute(fn, *args, ctx: IEEECtx):
    with gmp.context(
            precision=ctx.p,
            emin=ctx.emin-ctx.p+2,
            emax=ctx.emax+1,
            subnormalize=True,
            trap_underflow=False,
            trap_overflow=False,
            trap_inexact=False,
            trap_invalid=False,
            trap_erange=False,
            trap_divzero=False,
            round=gmp_rms[ctx.rm],
    ) as gmpctx:
        result = fn(*args)
        invalid = gmpctx.invalid
        divzero = gmpctx.divzero
        overflow = gmpctx.overflow
        underflow = gmpctx.underflow
        inexact = gmpctx.inexact

        # underflow should only be raised with inexact
        if underflow and not inexact:
            underflow = False

        # overflow
        # if gmpctx.overflow:
        #     result = gmp.inf(gmp.sign(result))

    return result, IEEEFlags(invalid, divzero, overflow, underflow, inexact)

def check_equality(x, y):
    xf, xflags = x
    yf, yflags = y

    xnan = xf.is_nan()
    ynan = yf.is_nan()
    if xnan != ynan:
        return False

    if xflags.divzero != yflags.divzero or \
       xflags.overflow != yflags.overflow or \
       xflags.underflow != yflags.underflow or \
       xflags.inexact != yflags.inexact or \
       xflags.invalid != yflags.invalid:
       return False

    return xnan and ynan or (gmp.is_signed(xf) == gmp.is_signed(yf) and xf == yf)
    

def test_exhaustive_1ary(op, es, nbits, rm):
    pass

def test_exhaustive_2ary(op, es, nbits, rm):
    ctx = ieee_ctx(es, nbits, rm)
    fl_fn = fl_ops[op]
    mpfr_fn = gmpmath.gmp_ops[op]
    for i in range(2 ** nbits):
        x = bits_to_digital(i, ctx)
        for j in range(2 ** nbits):
            y = bits_to_digital(j, ctx)
            # translate to MPFR
            xgmp = gmpmath.digital_to_mpfr(x)
            ygmp = gmpmath.digital_to_mpfr(y)
            # compute
            rf = fl_fn(x, y, ctx=ctx)
            r, flags = gmpmath.digital_to_mpfr(rf), rf.get_flags()
            rgmp, flagsgmp = mpfr_compute(mpfr_fn, xgmp, ygmp, ctx=ctx)
            # check equality
            if not check_equality((r, flags), (rgmp, flagsgmp)):
                inputs = (xgmp, ygmp)
                expected = (rgmp, flagsgmp.invalid, flagsgmp.divzero, flagsgmp.overflow, flagsgmp.underflow, flagsgmp.inexact)
                actual = (r, flags.invalid, flags.divzero, flags.overflow, flags.underflow, flags.inexact)
                print(' inputs={}, expected={}, actual={}'.format(inputs, expected, actual))


def test_exhaustive_3ary(op, es, nbits, rm):
    pass

def test_exhaustive_thread(config):
    op, es, nbits, rm = config
    argc = len(getargspec(fl_ops[op]).args) - 1
    print('test={}, es={}, nbits={}, rm={}'.format(repr(op), es, nbits, repr(rm)))

    if argc == 1:
        test_exhaustive_1ary(op, es, nbits, rm)
    elif argc == 2:
        test_exhaustive_2ary(op, es, nbits, rm)
    elif argc == 3:
        test_exhaustive_3ary(op, es, nbits, rm)
    else:
        raise ValueError('unexpected argument count: {}'.format(argc))

def test_exhaustive(es_min=2, nbits_min=4, es_max=8, nbits_max=10, threads=None):
    ops = [OP.add, OP.sub, OP.mul, OP.div]
    rms = [RM.RNE, RM.RTP, RM.RTN, RM.RTZ, RM.RAZ]

    configs = [] 
    for es in range(es_min, es_max + 1):
        for nbits in range(max(nbits_min, es + 2), nbits_max + 1):
            for op in ops:
                for rm in rms:
                    configs.append((op, es, nbits, rm))

    if threads is not None:
        with mp.Pool(processes=threads) as pool:
            pool.map(test_exhaustive_thread, configs)
    else:
        for config in configs:
            test_exhaustive_thread(config)

def test(threads=None):
    test_exhaustive(threads=threads)
