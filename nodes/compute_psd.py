from scipy import signal as sps

from gen.messages_pb2 import ComputePSDInput, PSDResult
from gen.axiom_context import AxiomContext
from nodes._common import err, validate_signal, to_pylist, VALID_WINDOWS


def compute_psd(ax: AxiomContext, input: ComputePSDInput) -> PSDResult:
    """Estimates the power spectral density of a signal via Welch's method
    (scipy.signal.welch — averaged periodograms over overlapping windows,
    lower variance) or a single periodogram (scipy.signal.periodogram —
    higher resolution, higher variance). Requires sample_rate_hz > 0.
    """
    values, rate, e = validate_signal(input.signal, min_len=2, require_rate=True)
    if e:
        return PSDResult(error=e)

    method = input.method or "welch"
    if method not in ("welch", "periodogram"):
        return PSDResult(error=err("INVALID_ARGUMENT", f"method must be 'welch' or 'periodogram', got '{method}'"))

    window = input.window or ("hann" if method == "welch" else "boxcar")
    if window not in VALID_WINDOWS - {"kaiser"}:
        return PSDResult(error=err("INVALID_ARGUMENT", f"window must be one of {sorted(VALID_WINDOWS - {'kaiser'})}, got '{window}'"))

    nperseg = input.nperseg if input.nperseg > 0 else None
    if nperseg is not None and nperseg > len(values):
        return PSDResult(error=err("INVALID_ARGUMENT", f"nperseg ({nperseg}) must not exceed the signal length ({len(values)})"))

    try:
        if method == "welch":
            freqs, psd = sps.welch(values, fs=rate, window=window, nperseg=nperseg)
        else:
            freqs, psd = sps.periodogram(values, fs=rate, window=window)
    except Exception as exc:
        return PSDResult(error=err("COMPUTE_ERROR", f"PSD estimation failed: {exc}"))

    return PSDResult(frequencies_hz=to_pylist(freqs), psd=to_pylist(psd))
