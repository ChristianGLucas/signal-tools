import numpy as np

from gen.messages_pb2 import SmoothInput, Signal
from nodes.smooth_signal import smooth_signal
from nodes._test_helpers import _TestContext


def test_smooth_signal_savgol_exactly_reproduces_a_polynomial_input():
    """Oracle: a Savitzky-Golay filter of polynomial order p exactly
    reproduces any input signal that is itself a polynomial of degree <= p
    (a defining mathematical property of the least-squares-polynomial
    filter, true regardless of scipy's specific implementation).
    """
    ax = _TestContext()
    ramp = [2.0 * i + 3.0 for i in range(50)]  # degree-1 polynomial
    result = smooth_signal(ax, SmoothInput(signal=Signal(values=ramp, sample_rate_hz=1.0), window_length=7, polyorder=2))
    assert result.error.code == ""
    assert np.allclose(result.signal.values, ramp, atol=1e-6)


def test_smooth_signal_reduces_noise_variance():
    rng = np.random.default_rng(42)
    t = np.linspace(0, 4 * np.pi, 200)
    clean = np.sin(t)
    noisy = clean + rng.normal(0, 0.3, size=200)

    ax = _TestContext()
    result = smooth_signal(ax, SmoothInput(signal=Signal(values=noisy.tolist(), sample_rate_hz=1.0), window_length=11, polyorder=3))
    assert result.error.code == ""

    smoothed = np.array(result.signal.values)
    assert np.std(smoothed - clean) < np.std(noisy - clean)


def test_smooth_signal_rejects_even_window_length():
    ax = _TestContext()
    result = smooth_signal(ax, SmoothInput(signal=Signal(values=[1.0, 2.0, 3.0, 4.0, 5.0], sample_rate_hz=1.0), window_length=4, polyorder=2))
    assert result.error.code == "INVALID_ARGUMENT"
