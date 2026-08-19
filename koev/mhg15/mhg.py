"""Python wrapper for Plamen Koev's mhg (hypergeometric function of a matrix
argument), via the standalone C core in mhg_core.c — no MATLAB/Octave needed.

Build the shared library first (done by build.sh):

    clang -O3 -fPIC -shared mhg_core.c -o libmhg.dylib

Usage mirrors the original MATLAB  [s,c] = mhg([MAX,K,lambda], alpha, p, q, x, [y]):

    from mhg import mhg
    s = mhg(30, 2, [], [], [0.1, 0.2])          # 0F0 == exp(sum(x))
    s, coef = mhg(40, 2, [3.0], [5.0], x, want_coef=True)

`arg0` may be a scalar MAX or a sequence [MAX, K, lambda_1, lambda_2, ...].
"""

import ctypes
import os
from collections.abc import Sequence

import numpy as np
import sys

lib_name = 'libmhg.dylib' if sys.platform == 'darwin' else 'libmhg.so'
_LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), lib_name)
_lib = ctypes.CDLL(_LIB)

_dptr = ctypes.POINTER(ctypes.c_double)
_lib.mhg_eval.restype = ctypes.c_int
_lib.mhg_eval.argtypes = [
    _dptr, ctypes.c_int,          # arg0, arg0len
    ctypes.c_double,              # alpha
    _dptr, ctypes.c_int,          # p, np
    _dptr, ctypes.c_int,          # q, nq
    _dptr, ctypes.c_int,          # x, n
    _dptr,                        # y (or NULL)
    _dptr, _dptr,                 # s_out, coef_out (or NULL)
    ctypes.c_char_p, ctypes.c_int # err, errlen
]


def _vec(a):
    """Contiguous float64 1-D array (empty allowed)."""
    return np.ascontiguousarray(np.atleast_1d(np.asarray(a, dtype=np.float64)).ravel())


def _ptr(arr):
    return arr.ctypes.data_as(_dptr) if arr.size else None


def mhg(arg0, alpha, p, q, x, y=None, want_coef=False):
    """Truncated hypergeometric function pFq^alpha(p; q; x [; y]) of a matrix
    argument (matrices given by their eigenvalue vectors x and, optionally, y).

    Parameters
    ----------
    arg0 : float or sequence
        MAX (truncation order), or [MAX, K, lambda_1, lambda_2, ...].
    alpha : float
        1 -> Schur functions, 2 -> zonal polynomials (the usual real case).
    p, q : sequences
        Numerator (a_i) and denominator (b_j) parameters. May be empty.
    x : sequence
        Eigenvalues of the matrix argument.
    y : sequence, optional
        Eigenvalues of a second matrix argument (same length as x).
    want_coef : bool
        If True, also return the per-degree coefficients (len MAX+1).

    Returns
    -------
    s            (float)                if want_coef is False
    (s, coef)    (float, np.ndarray)    if want_coef is True
    """
    arg0v = _vec([arg0] if not isinstance(arg0, Sequence) else arg0)
    if isinstance(arg0, (int, float)):
        arg0v = _vec([arg0])
    pv, qv, xv = _vec(p), _vec(q), _vec(x)

    if y is not None:
        yv = _vec(y)
        if yv.size != xv.size:
            raise ValueError("x and y must have the same length")
        yptr = _ptr(yv)
    else:
        yptr = None

    MAX = int(arg0v[0])
    s = ctypes.c_double(0.0)
    coef = np.zeros(MAX + 1, dtype=np.float64) if want_coef else None
    errbuf = ctypes.create_string_buffer(256)

    rc = _lib.mhg_eval(
        arg0v.ctypes.data_as(_dptr), arg0v.size,
        ctypes.c_double(float(alpha)),
        _ptr(pv), pv.size,
        _ptr(qv), qv.size,
        xv.ctypes.data_as(_dptr), xv.size,
        yptr,
        ctypes.byref(s),
        coef.ctypes.data_as(_dptr) if coef is not None else None,
        errbuf, len(errbuf),
    )
    if rc != 0:
        raise RuntimeError("mhg_eval: " + errbuf.value.decode(errors="replace"))

    return (s.value, coef) if want_coef else s.value


if __name__ == "__main__":
    # 0F0^alpha(x) = etr(X) = exp(sum(x)), independent of alpha
    x = [0.1, 0.2, 0.05]
    got = mhg(60, 2.0, [], [], x)
    print(f"0F0(x)      = {got:.12f}   exp(sum x) = {np.exp(np.sum(x)):.12f}")

    # 1F0^alpha(a;;X) = prod (1-x_i)^(-a)  (alpha=2, |x_i|<1)
    a = 3.0
    got = mhg(80, 2.0, [a], [], x)
    exact = np.prod([(1 - xi) ** (-a) for xi in x])
    print(f"1F0(a;;x)   = {got:.12f}   prod(1-x)^-a = {exact:.12f}")
