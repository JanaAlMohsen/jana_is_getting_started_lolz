"""
labcheck.py - self-checks for ELEC3901 lab tasks.

    from labcheck import check, try_it
    check('fk at test pose', lambda: my_fk(q_test), 'a3f9c1e2b0', decimals=3)      # a TASK: hidden reference
    try_it('fixed axes', R_fixed, (SO3.Ry(90, 'deg') * SO3.Rx(90, 'deg')).A)          # an EXERCISE: a reference you can see

The expected value is stored as a short hash of the rounded result, so a passing check tells you your
function agrees with the reference to the stated number of decimals without revealing the number.
Shapes and ordering matter: return exactly what the task's docstring asks for.
"""
import hashlib
import numpy as np


def digest(value, decimals=3):
    a = np.round(np.asarray(value, dtype=float), decimals) + 0.0     # +0.0 turns -0.0 into 0.0
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:10]


def check(label, value, expected, decimals=3):
    """value may be the result itself or a zero-argument callable (evaluated here so errors are caught)."""
    try:
        if callable(value):
            value = value()
        got = digest(value, decimals)
    except Exception as e:                      # NotImplementedError, wrong type, etc.
        print(f'  x  {label}: could not evaluate ({type(e).__name__}: {e})')
        return False
    ok = got == expected
    print(('  OK ' if ok else '  x  ') + label + ('' if ok else '   -> result does not match the reference (check shape, units, sign, rounding)'))
    return ok


def _arr(v):
    return np.asarray(getattr(v, 'A', v), dtype=float)


def try_it(label, value, expected, tol=1e-6):
    """Exercise check: compare your value with a reference computed another way (usually the toolbox), and say what looks wrong.
    value may be a zero-argument callable; a value still containing ... prints 'not done yet' instead of failing."""
    todo = lambda v: v is Ellipsis or v is None or (isinstance(v, (list, tuple)) and any(x is Ellipsis or x is None for x in v))  # noqa: E731
    try:
        if callable(value):
            value = value()
        if todo(value) or expected is Ellipsis:
            print(f'  ..  {label}: not done yet - replace the ... in the cell above (a function that returns nothing gives None)')
            return None
        a, b = _arr(value), _arr(expected)
    except NotImplementedError:
        print(f'  ..  {label}: not done yet - it uses a task function that is still TODO')
        return None
    except Exception as e:
        if 'ellipsis' in str(e).lower() or 'nonetype' in str(e).lower():
            print(f'  ..  {label}: not done yet - something it uses is still ...')
            return None
        print(f'  x  {label}: could not evaluate ({type(e).__name__}: {e})')
        return False
    vec = lambda x: x.ndim <= 1 or min(x.shape) == 1                    # noqa: E731  a vector, as a row, a column or flat
    if a.shape != b.shape and a.size == b.size and vec(a) and vec(b):
        a, b = a.ravel(), b.ravel()                                       # the same vector in another shape: compare the numbers
    if a.shape != b.shape:
        hint = '  (transposed?)' if a.T.shape == b.shape and np.allclose(a.T, b, atol=tol) else ''
        print(f'  x  {label}: shape {a.shape}, expected {b.shape}{hint}')
        return False
    if not np.all(np.isfinite(a)):
        print(f'  x  {label}: contains NaN or inf')
        return False
    if np.allclose(a, b, atol=tol):
        print(f'  OK {label}')
        return True
    if a.ndim == 2 and a.shape[0] == a.shape[1] and np.allclose(a.T, b, atol=tol):
        hint = 'transposed (rows and columns swapped) - or the inverse of a rotation?'
    elif np.allclose(-a, b, atol=tol):
        hint = 'the sign is flipped'
    else:
        hint = f'largest difference {np.abs(a - b).max():.3g}'
    print(f'  x  {label}: does not match - {hint}')
    return False
