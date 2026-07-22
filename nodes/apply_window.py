from scipy.signal import windows as spwin

from gen.messages_pb2 import ApplyWindowInput, SignalResult
from gen.axiom_context import AxiomContext
from nodes._common import err, validate_signal, signal_out


def apply_window(ax: AxiomContext, input: ApplyWindowInput) -> SignalResult:
    """Multiplies a signal by a window function (Hann, Hamming, Blackman, or
    Kaiser, via scipy.signal.windows) — commonly a pre-processing step before
    FFT/PSD analysis to reduce spectral leakage. sample_rate_hz is passed
    through unchanged.
    """
    values, rate, e = validate_signal(input.signal, min_len=1)
    if e:
        return SignalResult(error=e)

    window = input.window or "hann"
    n = len(values)
    try:
        if window == "hann":
            win = spwin.hann(n)
        elif window == "hamming":
            win = spwin.hamming(n)
        elif window == "blackman":
            win = spwin.blackman(n)
        elif window == "kaiser":
            beta = input.beta if input.beta > 0 else 8.6
            win = spwin.kaiser(n, beta)
        else:
            return SignalResult(error=err("INVALID_ARGUMENT", f"window must be one of 'hann', 'hamming', 'blackman', 'kaiser', got '{window}'"))
    except Exception as exc:
        return SignalResult(error=err("COMPUTE_ERROR", f"window generation failed: {exc}"))

    return SignalResult(signal=signal_out(values * win, rate))
