"""Shared helpers for christiangeorgelucas/signal-tools nodes.

Centralizes structured Error construction, Signal <-> numpy conversion, and
the finiteness/argument validation every node enforces on caller-controlled
input before any scipy call.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

from gen.messages_pb2 import Error, FilterCoefficients, Signal

VALID_DESIGNS = {"butterworth": "butter", "cheby1": "cheby1", "cheby2": "cheby2", "bessel": "bessel", "elliptic": "ellip"}
VALID_BAND_TYPES = {"lowpass", "highpass", "bandpass", "bandstop"}
VALID_WINDOWS = {"hann", "hamming", "blackman", "bartlett", "boxcar", "kaiser"}


def err(code: str, message: str) -> Error:
    """Build a structured Error. Codes: INVALID_INPUT (empty/too-short
    signal), INVALID_ARGUMENT (bad parameter), COMPUTE_ERROR (the
    underlying scipy routine could not produce a result for this data)."""
    return Error(code=code, message=message)


def to_array(signal: Signal) -> np.ndarray:
    return np.asarray(signal.values, dtype=np.float64)


def validate_signal(signal: Signal, *, min_len: int = 1, require_rate: bool = False, name: str = "signal"):
    """Validate a Signal against the package's bounds. Returns
    (values_ndarray, sample_rate_hz, Error|None); on error the first two are
    None."""
    values = to_array(signal)
    n = values.size
    if n < min_len:
        return None, None, err("INVALID_INPUT", f"{name} needs at least {min_len} sample(s), got {n}")
    if n and not np.all(np.isfinite(values)):
        return None, None, err("INVALID_INPUT", f"{name} contains NaN or infinite values")
    rate = signal.sample_rate_hz
    if require_rate and rate <= 0.0:
        return None, None, err("INVALID_ARGUMENT", f"{name}.sample_rate_hz must be > 0 for this operation")
    if rate < 0.0:
        return None, None, err("INVALID_ARGUMENT", f"{name}.sample_rate_hz must not be negative")
    return values, rate, None


def signal_out(values: np.ndarray, sample_rate_hz: float) -> Signal:
    return Signal(values=to_pylist(values), sample_rate_hz=float(sample_rate_hz))


def to_pylist(arr) -> list:
    """Convert a numpy array (or any iterable of numpy/py scalars) to a
    plain list of native Python floats, which is what a repeated `double`
    field needs."""
    return [float(x) for x in np.asarray(arr, dtype=np.float64)]


def to_pyintlist(arr) -> list:
    return [int(x) for x in np.asarray(arr)]


def validate_design_params(design: str, band_type: str, order: int, cutoff_hz, sample_rate_hz: float, ripple_db: float, stopband_atten_db: float):
    """Validate the common filter-design parameter set shared by
    DesignFilter and FilterSignal. Returns (scipy_ftype, scipy_btype,
    cutoffs_list, Error|None)."""
    design = design or "butterworth"
    band_type = band_type or "lowpass"
    if design not in VALID_DESIGNS:
        return None, None, None, err("INVALID_ARGUMENT", f"design must be one of {sorted(VALID_DESIGNS)}, got '{design}'")
    if band_type not in VALID_BAND_TYPES:
        return None, None, None, err("INVALID_ARGUMENT", f"band_type must be one of {sorted(VALID_BAND_TYPES)}, got '{band_type}'")
    if order < 1:
        return None, None, None, err("INVALID_ARGUMENT", f"order must be >= 1, got {order}")
    if order > 20:
        return None, None, None, err("LIMIT_EXCEEDED", f"order must be <= 20 (higher orders are numerically unstable in this form), got {order}")
    if sample_rate_hz <= 0.0:
        return None, None, None, err("INVALID_ARGUMENT", "sample_rate_hz must be > 0")

    cutoffs = list(cutoff_hz)
    expected_n = 1 if band_type in ("lowpass", "highpass") else 2
    if len(cutoffs) != expected_n:
        return None, None, None, err("INVALID_ARGUMENT", f"band_type '{band_type}' requires {expected_n} cutoff_hz value(s), got {len(cutoffs)}")
    nyquist = sample_rate_hz / 2.0
    for c in cutoffs:
        if not (0.0 < c < nyquist):
            return None, None, None, err("INVALID_ARGUMENT", f"cutoff_hz values must be in (0, {nyquist}) [the Nyquist frequency], got {c}")
    if expected_n == 2 and cutoffs[0] >= cutoffs[1]:
        return None, None, None, err("INVALID_ARGUMENT", f"cutoff_hz must be [low, high] with low < high, got {cutoffs}")

    if design in ("cheby1", "elliptic") and ripple_db <= 0.0:
        return None, None, None, err("INVALID_ARGUMENT", f"design '{design}' requires ripple_db > 0")
    if design in ("cheby2", "elliptic") and stopband_atten_db <= 0.0:
        return None, None, None, err("INVALID_ARGUMENT", f"design '{design}' requires stopband_atten_db > 0")

    return VALID_DESIGNS[design], band_type, cutoffs, None


def build_filter_sos(ftype: str, btype: str, order: int, cutoffs: list, fs: float, ripple_db: float, stopband_atten_db: float) -> np.ndarray:
    """Design a digital IIR filter with scipy.signal.iirfilter and return its
    second-order-sections representation. Callers must validate parameters
    with validate_design_params first."""
    wn = cutoffs[0] if len(cutoffs) == 1 else cutoffs
    kwargs = {}
    if ftype in ("cheby1", "ellip"):
        kwargs["rp"] = ripple_db
    if ftype in ("cheby2", "ellip"):
        kwargs["rs"] = stopband_atten_db
    return sps.iirfilter(order, wn, btype=btype, analog=False, ftype=ftype, fs=fs, output="sos", **kwargs)


def sos_to_coefficients(sos: np.ndarray, sample_rate_hz: float, output_form: str = "sos") -> FilterCoefficients:
    """Package an sos ndarray as a FilterCoefficients message, converting to
    transfer-function b/a form if requested."""
    if output_form == "ba":
        b, a = sps.sos2tf(sos)
        return FilterCoefficients(b=to_pylist(b), a=to_pylist(a), form="ba", sample_rate_hz=float(sample_rate_hz))
    return FilterCoefficients(sos=to_pylist(sos.flatten()), form="sos", sample_rate_hz=float(sample_rate_hz))


def coefficients_to_ba(coefficients: FilterCoefficients):
    """Return (b, a) ndarrays from a FilterCoefficients message, regardless
    of which form it carries."""
    if coefficients.form == "ba":
        return np.asarray(coefficients.b, dtype=np.float64), np.asarray(coefficients.a, dtype=np.float64)
    sos = np.asarray(coefficients.sos, dtype=np.float64).reshape(-1, 6)
    return sps.sos2tf(sos)


def coefficients_to_sos(coefficients: FilterCoefficients) -> np.ndarray:
    """Return an sos ndarray from a FilterCoefficients message, regardless of
    which form it carries."""
    if coefficients.form == "sos":
        return np.asarray(coefficients.sos, dtype=np.float64).reshape(-1, 6)
    b = np.asarray(coefficients.b, dtype=np.float64)
    a = np.asarray(coefficients.a, dtype=np.float64)
    return sps.tf2sos(b, a)
