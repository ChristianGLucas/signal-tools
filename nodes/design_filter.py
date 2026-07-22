from gen.messages_pb2 import DesignFilterInput, DesignFilterResult
from gen.axiom_context import AxiomContext
from nodes._common import err, validate_design_params, build_filter_sos, sos_to_coefficients


def design_filter(ax: AxiomContext, input: DesignFilterInput) -> DesignFilterResult:
    """Designs a digital IIR filter (Butterworth, Chebyshev I/II, Bessel, or
    elliptic; lowpass/highpass/bandpass/bandstop) via scipy.signal.iirfilter
    and returns its coefficients as second-order sections (default,
    numerically preferred) or transfer-function b/a, without applying it to
    any signal. Rejects a cutoff at or above the Nyquist frequency, a
    non-positive order, or a missing ripple/attenuation parameter for a
    design that requires one.
    """
    ftype, btype, cutoffs, e = validate_design_params(
        input.design, input.band_type, input.order, input.cutoff_hz,
        input.sample_rate_hz, input.ripple_db, input.stopband_atten_db,
    )
    if e:
        return DesignFilterResult(error=e)

    output_form = input.output_form or "sos"
    if output_form not in ("sos", "ba"):
        return DesignFilterResult(error=err("INVALID_ARGUMENT", f"output_form must be 'sos' or 'ba', got '{output_form}'"))

    try:
        sos = build_filter_sos(ftype, btype, input.order, cutoffs, input.sample_rate_hz, input.ripple_db, input.stopband_atten_db)
    except Exception as exc:  # scipy raises assorted exceptions on numerically-infeasible designs
        return DesignFilterResult(error=err("COMPUTE_ERROR", f"filter design failed: {exc}"))

    return DesignFilterResult(coefficients=sos_to_coefficients(sos, input.sample_rate_hz, output_form))
