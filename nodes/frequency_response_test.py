import numpy as np

from gen.messages_pb2 import FrequencyResponseInput, DesignFilterInput, FilterCoefficients
from nodes.frequency_response import frequency_response
from nodes.design_filter import design_filter
from nodes._test_helpers import _TestContext


def test_frequency_response_butterworth_is_minus_3db_at_cutoff():
    """Oracle: a Butterworth filter's magnitude response at its own cutoff
    frequency is, by the closed-form Butterworth design equation,
    -3.0103 dB (|H| = 1/sqrt(2)) — independent of order or this node's own
    code.
    """
    ax = _TestContext()
    coeffs = design_filter(ax, DesignFilterInput(
        design="butterworth", band_type="lowpass", order=4,
        cutoff_hz=[100.0], sample_rate_hz=1000.0,
    )).coefficients

    result = frequency_response(ax, FrequencyResponseInput(coefficients=coeffs, n_points=2000))
    assert result.error.code == ""

    freqs = np.array(result.frequencies_hz)
    idx = int(np.argmin(np.abs(freqs - 100.0)))
    assert abs(result.magnitude_db[idx] - (-3.0103)) < 0.15

    # Below cutoff must pass with near-0 dB attenuation; well above cutoff
    # must be attenuated far more than at the cutoff itself.
    idx_low = int(np.argmin(np.abs(freqs - 10.0)))
    idx_high = int(np.argmin(np.abs(freqs - 400.0)))
    assert result.magnitude_db[idx_low] > -1.0
    assert result.magnitude_db[idx_high] < result.magnitude_db[idx] - 20.0


def test_frequency_response_rejects_missing_sample_rate():
    ax = _TestContext()
    result = frequency_response(ax, FrequencyResponseInput(
        coefficients=FilterCoefficients(sos=[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], form="sos"),
    ))
    assert result.error.code == "INVALID_ARGUMENT"
