from gen.messages_pb2 import FilterSignalInput, SignalResult
from gen.axiom_context import AxiomContext
from nodes._common import err, validate_signal, validate_design_params, build_filter_sos, signal_out
from scipy import signal as sps


def filter_signal(ax: AxiomContext, input: FilterSignalInput) -> SignalResult:
    """Designs a Butterworth/Chebyshev-I/Chebyshev-II/Bessel/elliptic
    lowpass, highpass, bandpass, or bandstop filter and applies it to a
    signal in a single call (design defaults to Butterworth order 4) — the
    convenience path when you don't need the filter's coefficients
    themselves (use DesignFilter + ApplyFilter, or FrequencyResponse, for
    that).
    """
    values, rate, e = validate_signal(input.signal, min_len=1, require_rate=True)
    if e:
        return SignalResult(error=e)

    order = input.order or 4
    ftype, btype, cutoffs, e = validate_design_params(
        input.design, input.band_type, order, input.cutoff_hz,
        rate, input.ripple_db, input.stopband_atten_db,
    )
    if e:
        return SignalResult(error=e)

    mode = input.mode or "filtfilt"
    if mode not in ("filtfilt", "lfilter"):
        return SignalResult(error=err("INVALID_ARGUMENT", f"mode must be 'filtfilt' or 'lfilter', got '{mode}'"))

    try:
        sos = build_filter_sos(ftype, btype, order, cutoffs, rate, input.ripple_db, input.stopband_atten_db)
        filtered = sps.sosfiltfilt(sos, values) if mode == "filtfilt" else sps.sosfilt(sos, values)
    except Exception as exc:
        return SignalResult(error=err("COMPUTE_ERROR", f"design/apply failed: {exc}"))

    return SignalResult(signal=signal_out(filtered, rate))
