import numpy as np

from gen.messages_pb2 import RMSEnvelopeInput, SignalResult
from gen.axiom_context import AxiomContext
from nodes._common import err, validate_signal, signal_out


def compute_rms_envelope(ax: AxiomContext, input: RMSEnvelopeInput) -> SignalResult:
    """Computes the short-time RMS (root-mean-square) energy envelope of a
    signal over sliding, optionally-overlapping windows — a simple, robust
    measure of a signal's local energy over time. hop_size defaults to
    window_size (non-overlapping windows); the output's sample_rate_hz is
    the input's rate divided by hop_size.
    """
    values, rate, e = validate_signal(input.signal, min_len=1)
    if e:
        return SignalResult(error=e)

    window_size = input.window_size
    if window_size < 1:
        return SignalResult(error=err("INVALID_ARGUMENT", f"window_size must be >= 1, got {window_size}"))
    hop_size = input.hop_size if input.hop_size > 0 else window_size
    if hop_size < 1:
        return SignalResult(error=err("INVALID_ARGUMENT", f"hop_size must be >= 1, got {hop_size}"))
    if window_size > len(values):
        return SignalResult(error=err("INVALID_INPUT", f"window_size ({window_size}) must not exceed the signal length ({len(values)})"))

    # O(n) sliding-window sum-of-squares via a zero-prefixed cumulative sum,
    # so an arbitrarily small hop_size cannot make this quadratic.
    squared = values.astype(np.float64) ** 2
    cumsum = np.concatenate(([0.0], np.cumsum(squared)))
    starts = np.arange(0, len(values) - window_size + 1, hop_size)
    sums = cumsum[starts + window_size] - cumsum[starts]
    rms = np.sqrt(sums / window_size)

    new_rate = rate / hop_size if rate > 0 else 0.0
    return SignalResult(signal=signal_out(rms, new_rate))
