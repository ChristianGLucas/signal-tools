from scipy import signal as sps

from gen.messages_pb2 import TwoSignalsInput, SignalResult
from gen.axiom_context import AxiomContext
from nodes._common import err, validate_signal, signal_out


def convolve(ax: AxiomContext, input: TwoSignalsInput) -> SignalResult:
    """Convolves two signals via scipy.signal.fftconvolve, with the standard
    "full" (default; length len(a)+len(b)-1), "same" (centered, length
    max(len(a),len(b))), or "valid" (fully-overlapping samples only) output
    modes. The output's sample_rate_hz is taken from signal_a.
    """
    a, rate, e = validate_signal(input.signal_a, min_len=1, name="signal_a")
    if e:
        return SignalResult(error=e)
    b, _, e = validate_signal(input.signal_b, min_len=1, name="signal_b")
    if e:
        return SignalResult(error=e)

    mode = input.convolve_mode or "full"
    if mode not in ("full", "same", "valid"):
        return SignalResult(error=err("INVALID_ARGUMENT", f"convolve_mode must be 'full', 'same', or 'valid', got '{mode}'"))
    if mode == "valid" and min(len(a), len(b)) > max(len(a), len(b)):
        return SignalResult(error=err("INVALID_ARGUMENT", "mode 'valid' requires neither signal to be longer than the other in a way that yields no overlap"))

    try:
        result = sps.fftconvolve(a, b, mode=mode)
    except Exception as exc:
        return SignalResult(error=err("COMPUTE_ERROR", f"convolution failed: {exc}"))

    return SignalResult(signal=signal_out(result, rate))
