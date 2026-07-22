from scipy import signal as sps

from gen.messages_pb2 import SmoothInput, SignalResult
from gen.axiom_context import AxiomContext
from nodes._common import err, validate_signal, signal_out


def smooth_signal(ax: AxiomContext, input: SmoothInput) -> SignalResult:
    """Smooths a signal with a Savitzky-Golay filter
    (scipy.signal.savgol_filter) — a local polynomial least-squares fit over
    a sliding window — which preserves peak height and width better than a
    moving average. window_length must be odd, >= 3, greater than
    polyorder, and no larger than the signal's length.
    """
    values, rate, e = validate_signal(input.signal, min_len=3)
    if e:
        return SignalResult(error=e)

    window_length = input.window_length
    polyorder = input.polyorder
    if window_length < 3 or window_length % 2 == 0:
        return SignalResult(error=err("INVALID_ARGUMENT", f"window_length must be an odd integer >= 3, got {window_length}"))
    if window_length > len(values):
        return SignalResult(error=err("INVALID_ARGUMENT", f"window_length ({window_length}) must not exceed the signal length ({len(values)})"))
    if polyorder < 1:
        return SignalResult(error=err("INVALID_ARGUMENT", f"polyorder must be >= 1, got {polyorder}"))
    if polyorder >= window_length:
        return SignalResult(error=err("INVALID_ARGUMENT", f"polyorder ({polyorder}) must be less than window_length ({window_length})"))

    try:
        smoothed = sps.savgol_filter(values, window_length, polyorder)
    except Exception as exc:
        return SignalResult(error=err("COMPUTE_ERROR", f"smoothing failed: {exc}"))

    return SignalResult(signal=signal_out(smoothed, rate))
