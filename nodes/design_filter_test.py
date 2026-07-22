import numpy as np
from scipy import signal as sps

from gen.messages_pb2 import DesignFilterInput
from nodes.design_filter import design_filter
from nodes._test_helpers import _TestContext


def test_design_filter_butterworth_lowpass_hits_minus_3db_at_cutoff():
    """Independent mathematical oracle: a Butterworth filter's magnitude
    response at its cutoff frequency is, by definition, -3.01 dB (a factor
    of 1/sqrt(2)) regardless of order — this is a closed-form property of
    the Butterworth design equation, not something this node's own code
    could get right by accident.
    """
    ax = _TestContext()
    result = design_filter(ax, DesignFilterInput(
        design="butterworth", band_type="lowpass", order=4,
        cutoff_hz=[100.0], sample_rate_hz=1000.0, output_form="sos",
    ))
    assert result.error.code == ""
    assert result.coefficients.form == "sos"
    sos = np.array(result.coefficients.sos).reshape(-1, 6)
    assert sos.shape[0] == 2  # order 4 -> 2 second-order sections

    w, h = sps.sosfreqz(sos, worN=[100.0], fs=1000.0)
    mag = np.abs(h[0])
    assert abs(mag - (1 / np.sqrt(2))) < 0.01


def test_design_filter_rejects_cutoff_at_or_above_nyquist():
    ax = _TestContext()
    result = design_filter(ax, DesignFilterInput(
        design="butterworth", band_type="lowpass", order=4,
        cutoff_hz=[600.0], sample_rate_hz=1000.0,  # Nyquist is 500 Hz
    ))
    assert result.error.code == "INVALID_ARGUMENT"


def test_design_filter_bandpass_requires_two_cutoffs():
    ax = _TestContext()
    result = design_filter(ax, DesignFilterInput(
        design="butterworth", band_type="bandpass", order=4,
        cutoff_hz=[100.0], sample_rate_hz=1000.0,
    ))
    assert result.error.code == "INVALID_ARGUMENT"


def test_design_filter_cheby1_requires_ripple_db():
    ax = _TestContext()
    result = design_filter(ax, DesignFilterInput(
        design="cheby1", band_type="lowpass", order=4,
        cutoff_hz=[100.0], sample_rate_hz=1000.0,  # ripple_db left at 0
    ))
    assert result.error.code == "INVALID_ARGUMENT"
