import numpy as np
from scipy import signal as sps

from gen.messages_pb2 import FrequencyResponseInput, FrequencyResponseResult
from gen.axiom_context import AxiomContext
from nodes._common import err, to_pylist, coefficients_to_sos, coefficients_to_ba


def frequency_response(ax: AxiomContext, input: FrequencyResponseInput) -> FrequencyResponseResult:
    """Computes a digital filter's frequency response — magnitude (dB),
    phase (radians), and group delay (samples), evaluated at n_points
    frequencies from 0 to Nyquist — via scipy.signal.sosfreqz/freqz and
    scipy.signal.group_delay, for analyzing a filter's behavior without
    applying it to any data.
    """
    coeffs = input.coefficients
    if coeffs.form not in ("sos", "ba"):
        return FrequencyResponseResult(error=err("INVALID_ARGUMENT", f"coefficients.form must be 'sos' or 'ba', got '{coeffs.form}'"))

    fs = input.sample_rate_hz if input.sample_rate_hz > 0 else coeffs.sample_rate_hz
    if fs <= 0:
        return FrequencyResponseResult(error=err("INVALID_ARGUMENT", "sample_rate_hz must be > 0 (set it on the request or carried on coefficients)"))

    n_points = input.n_points if input.n_points > 0 else 512

    try:
        if coeffs.form == "sos":
            sos = coefficients_to_sos(coeffs)
            w, h = sps.sosfreqz(sos, worN=n_points, fs=fs)
        else:
            b, a = coefficients_to_ba(coeffs)
            w, h = sps.freqz(b, a, worN=n_points, fs=fs)

        b_ba, a_ba = coefficients_to_ba(coeffs)
        w_gd, gd = sps.group_delay((b_ba, a_ba), w=n_points, fs=fs)
    except Exception as exc:
        return FrequencyResponseResult(error=err("COMPUTE_ERROR", f"frequency-response computation failed: {exc}"))

    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(h), 1e-300))
    phase = np.angle(h)

    return FrequencyResponseResult(
        frequencies_hz=to_pylist(w),
        magnitude_db=to_pylist(magnitude_db),
        phase_rad=to_pylist(phase),
        group_delay_samples=to_pylist(gd),
    )
