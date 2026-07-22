import numpy as np
from scipy import signal as sps

from gen.messages_pb2 import CrossCorrelateInput, CrossCorrelateResult
from gen.axiom_context import AxiomContext
from nodes._common import err, validate_signal, to_pylist, to_pyintlist


def cross_correlate(ax: AxiomContext, input: CrossCorrelateInput) -> CrossCorrelateResult:
    """Cross-correlates two signals via scipy.signal.correlate, returning
    the correlation value and integer lag (in samples) for each output
    point; optionally normalized so a signal's autocorrelation at zero lag
    is 1.0. Leave signal_b empty (or pass the same signal twice) to compute
    an autocorrelation.
    """
    a, rate, e = validate_signal(input.signal_a, min_len=1, name="signal_a")
    if e:
        return CrossCorrelateResult(error=e)

    if len(input.signal_b.values) == 0:
        b = a
    else:
        b, _, e = validate_signal(input.signal_b, min_len=1, name="signal_b")
        if e:
            return CrossCorrelateResult(error=e)

    mode = input.correlate_mode or "full"
    if mode not in ("full", "same", "valid"):
        return CrossCorrelateResult(error=err("INVALID_ARGUMENT", f"correlate_mode must be 'full', 'same', or 'valid', got '{mode}'"))

    try:
        corr = sps.correlate(a, b, mode=mode)
        lags = sps.correlation_lags(len(a), len(b), mode=mode)
    except Exception as exc:
        return CrossCorrelateResult(error=err("COMPUTE_ERROR", f"cross-correlation failed: {exc}"))

    if input.normalize:
        norm = np.sqrt(np.sum(a.astype(np.float64) ** 2) * np.sum(np.asarray(b, dtype=np.float64) ** 2))
        if norm <= 0:
            return CrossCorrelateResult(error=err("COMPUTE_ERROR", "cannot normalize: one or both signals have zero energy"))
        corr = corr / norm

    return CrossCorrelateResult(correlation=to_pylist(corr), lags=to_pyintlist(lags))
