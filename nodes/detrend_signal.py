from scipy import signal as sps

from gen.messages_pb2 import DetrendInput, SignalResult
from gen.axiom_context import AxiomContext
from nodes._common import err, validate_signal, signal_out


def detrend_signal(ax: AxiomContext, input: DetrendInput) -> SignalResult:
    """Removes a linear (best-fit line) or constant (mean) trend from a
    signal via scipy.signal.detrend. sample_rate_hz is passed through
    unchanged.
    """
    values, rate, e = validate_signal(input.signal, min_len=1)
    if e:
        return SignalResult(error=e)

    detrend_type = input.detrend_type or "linear"
    if detrend_type not in ("linear", "constant"):
        return SignalResult(error=err("INVALID_ARGUMENT", f"detrend_type must be 'linear' or 'constant', got '{detrend_type}'"))

    try:
        detrended = sps.detrend(values, type=detrend_type)
    except Exception as exc:
        return SignalResult(error=err("COMPUTE_ERROR", f"detrend failed: {exc}"))

    return SignalResult(signal=signal_out(detrended, rate))
