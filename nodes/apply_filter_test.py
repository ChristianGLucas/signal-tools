import numpy as np

from gen.messages_pb2 import ApplyFilterInput, DesignFilterInput
from nodes.apply_filter import apply_filter
from nodes.design_filter import design_filter
from nodes._test_helpers import _TestContext, two_tone_signal, FS, N, BIN1, BIN2


def _rfft_mag(values):
    return np.abs(np.fft.rfft(np.asarray(values)))


def test_apply_filter_filtfilt_lowpass_isolates_the_low_tone():
    """Oracle: filtering a signal containing 50Hz+120Hz components through a
    lowpass filter with cutoff between them must leave the 50Hz energy
    intact while suppressing the 120Hz energy — verified independently via
    numpy.fft (not this package's ComputeFFT node).
    """
    ax = _TestContext()
    coeffs = design_filter(ax, DesignFilterInput(
        design="butterworth", band_type="lowpass", order=6,
        cutoff_hz=[80.0], sample_rate_hz=FS,
    )).coefficients

    result = apply_filter(ax, ApplyFilterInput(signal=two_tone_signal(), coefficients=coeffs, mode="filtfilt"))
    assert result.error.code == ""
    assert len(result.signal.values) == N

    mag = _rfft_mag(result.signal.values)
    assert mag[BIN1] > 800.0
    assert mag[BIN2] < 0.05 * mag[BIN1]


def test_apply_filter_rejects_unknown_coefficient_form():
    ax = _TestContext()
    result = apply_filter(ax, ApplyFilterInput(signal=two_tone_signal()))
    assert result.error.code == "INVALID_ARGUMENT"
