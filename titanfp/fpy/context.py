"""Rounding contexts in the FPY language"""

from typing import Any, Optional, Self

from .utils import raise_type_error

def current_context():
    """Returns the current global rounding context."""
    return _current_context

class RoundingContext(object):
    """Rounding contexts in FPY: implemented as a context manager"""
    _prec: Optional[str]
    _round: Optional[str]
    _kwds: dict[str, Any]
    _prev: Optional[Self]

    def __init__(
        self,
        ctx: Optional[Self] = None,
        prec: Optional[str] = None,
        round: Optional[str] = None,
        **kwargs):

        if ctx is not None:
            if not isinstance(ctx, RoundingContext):
                raise_type_error(RoundingContext, ctx)
            self._prec = ctx._prec
            self._round = ctx._round
            self._kwds = ctx._kwds
            self._prev = ctx._prev
        else:
            self._prec = None
            self._round = None
            self._kwds = {}
            self._prev = None

        if prec is not None:
            if not isinstance(prec, str):
                raise_type_error(str, prec)
            self._prec = prec

        if round is not None:
            if not isinstance(round, str):
                raise_type_error(str, round)
            self._round = round

        if kwargs is not None:
            if not isinstance(kwargs, dict):
                raise_type_error(dict, prec)
            self._kwds = { **self._kwds, **kwargs }

    def __repr__(self):
        name = self.__class__.__name__
        items = (f'{k}={v}' for k, v in self.__dict__.items())
        return f'{name}({', '.join(items)})'

    def __enter__(self):
        global _current_context
        self._prev = _current_context
        _current_context = self
        return self
    
    def __exit__(self, *_):
        global _current_context
        if _current_context is None:
            raise RuntimeError('no previous context')
        _current_context = self._prev
    

_current_context: Optional[RoundingContext] = None
"""Current global rounding context. DO NOT DIRECTLY ACCESS"""
