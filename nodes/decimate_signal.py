from scipy import signal as sps

from gen.messages_pb2 import DecimateInput, SignalResult
from gen.axiom_context import AxiomContext
from nodes._common import err, validate_signal, signal_out


def decimate_signal(ax: AxiomContext, input: DecimateInput) -> SignalResult:
    """Downsamples a signal by an integer factor with an anti-aliasing
    low-pass filter applied before subsampling (scipy.signal.decimate),
    preventing the aliasing that naive stride-based downsampling introduces.
    """
    values, rate, e = validate_signal(input.signal, min_len=2)
    if e:
        return SignalResult(error=e)

    factor = input.factor
    if factor < 2:
        return SignalResult(error=err("INVALID_ARGUMENT", f"factor must be >= 2, got {factor}"))
    if len(values) <= factor:
        return SignalResult(error=err("INVALID_INPUT", f"signal length ({len(values)}) must exceed factor ({factor})"))

    try:
        decimated = sps.decimate(values, factor)
    except Exception as exc:
        return SignalResult(error=err("COMPUTE_ERROR", f"decimation failed: {exc}"))

    new_rate = rate / factor if rate > 0 else 0.0
    return SignalResult(signal=signal_out(decimated, new_rate))
