from gen.messages_pb2 import ApplyFilterInput, SignalResult
from gen.axiom_context import AxiomContext
from nodes._common import err, validate_signal, signal_out, coefficients_to_sos, coefficients_to_ba
from scipy import signal as sps


def apply_filter(ax: AxiomContext, input: ApplyFilterInput) -> SignalResult:
    """Applies previously-designed filter coefficients (second-order
    sections or transfer-function b/a, e.g. from DesignFilter) to a signal —
    either zero-phase (scipy.signal.sosfiltfilt/filtfilt, forward+backward,
    no phase distortion) or single-pass causal (scipy.signal.sosfilt/lfilter,
    introduces phase delay).
    """
    values, rate, e = validate_signal(input.signal, min_len=1)
    if e:
        return SignalResult(error=e)

    coeffs = input.coefficients
    if coeffs.form not in ("sos", "ba"):
        return SignalResult(error=err("INVALID_ARGUMENT", f"coefficients.form must be 'sos' or 'ba', got '{coeffs.form}'"))

    mode = input.mode or "filtfilt"
    if mode not in ("filtfilt", "lfilter"):
        return SignalResult(error=err("INVALID_ARGUMENT", f"mode must be 'filtfilt' or 'lfilter', got '{mode}'"))

    try:
        if coeffs.form == "sos":
            sos = coefficients_to_sos(coeffs)
            filtered = sps.sosfiltfilt(sos, values) if mode == "filtfilt" else sps.sosfilt(sos, values)
        else:
            b, a = coefficients_to_ba(coeffs)
            filtered = sps.filtfilt(b, a, values) if mode == "filtfilt" else sps.lfilter(b, a, values)
    except ValueError as exc:
        # e.g. signal too short for filtfilt's default padlen given this filter's order
        return SignalResult(error=err("COMPUTE_ERROR", f"filtering failed: {exc}"))

    return SignalResult(signal=signal_out(filtered, rate))
