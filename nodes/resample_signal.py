from scipy import signal as sps

from gen.messages_pb2 import ResampleInput, SignalResult
from gen.axiom_context import AxiomContext
from nodes._common import err, validate_signal, signal_out


def resample_signal(ax: AxiomContext, input: ResampleInput) -> SignalResult:
    """Resamples a signal to a target number of samples using FFT-based
    resampling (scipy.signal.resample) — resizes the array while
    preserving the signal's frequency content. Distinct from
    DecimateSignal's integer-factor anti-aliased downsampling. If the input
    signal's sample_rate_hz is set, the output's is rescaled proportionally.
    """
    values, rate, e = validate_signal(input.signal, min_len=2)
    if e:
        return SignalResult(error=e)

    target_length = input.target_length
    if target_length < 1:
        return SignalResult(error=err("INVALID_ARGUMENT", f"target_length must be >= 1, got {target_length}"))

    try:
        resampled = sps.resample(values, target_length)
    except Exception as exc:
        return SignalResult(error=err("COMPUTE_ERROR", f"resampling failed: {exc}"))

    new_rate = rate * (target_length / len(values)) if rate > 0 else 0.0
    return SignalResult(signal=signal_out(resampled, new_rate))
