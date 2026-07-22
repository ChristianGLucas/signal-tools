import numpy as np

from gen.messages_pb2 import CrossCorrelateInput, Signal
from nodes.cross_correlate import cross_correlate
from nodes._test_helpers import _TestContext, two_tone_signal


def test_cross_correlate_normalized_autocorrelation_peaks_at_1_at_zero_lag():
    """Oracle: normalized autocorrelation of ANY nonzero signal with itself
    is, by definition, exactly 1.0 at zero lag, and 1.0 is the global
    maximum (Cauchy-Schwarz) — a mathematical identity independent of this
    node's implementation.
    """
    ax = _TestContext()
    result = cross_correlate(ax, CrossCorrelateInput(signal_a=two_tone_signal(), normalize=True))
    assert result.error.code == ""

    lags = list(result.lags)
    corr = list(result.correlation)
    zero_idx = lags.index(0)
    assert abs(corr[zero_idx] - 1.0) < 1e-6
    assert corr[zero_idx] == max(corr)


def test_cross_correlate_finds_known_impulse_shift():
    """Oracle: correlating an impulse at index 2 against an impulse at index
    5 (both in a length-8 zero signal) peaks at lag = 5 - 2 = 3, a
    hand-computed fact about impulse cross-correlation.
    """
    ax = _TestContext()
    a = Signal(values=[0.0] * 8, sample_rate_hz=1.0)
    a.values[2] = 1.0
    b = Signal(values=[0.0] * 8, sample_rate_hz=1.0)
    b.values[5] = 1.0

    result = cross_correlate(ax, CrossCorrelateInput(signal_a=a, signal_b=b, correlate_mode="full"))
    assert result.error.code == ""
    lags = list(result.lags)
    corr = list(result.correlation)
    peak_lag = lags[int(np.argmax(corr))]
    assert peak_lag == -3


def test_cross_correlate_normalize_of_zero_energy_signal_is_a_structured_error():
    ax = _TestContext()
    result = cross_correlate(ax, CrossCorrelateInput(
        signal_a=Signal(values=[0.0, 0.0, 0.0], sample_rate_hz=1.0), normalize=True,
    ))
    assert result.error.code == "COMPUTE_ERROR"
