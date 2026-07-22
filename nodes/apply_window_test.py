import numpy as np

from gen.messages_pb2 import ApplyWindowInput, Signal
from nodes.apply_window import apply_window
from nodes._test_helpers import _TestContext


def test_apply_window_hann_matches_the_closed_form_hann_curve():
    """Oracle: the Hann window has the closed-form definition
    w[n] = 0.5 - 0.5*cos(2*pi*n/(N-1)) — computed here from scratch (not via
    scipy) and compared to the node's output on an all-ones input signal
    (so the output IS the window curve).
    """
    n = 64
    ax = _TestContext()
    ones = Signal(values=[1.0] * n, sample_rate_hz=100.0)
    result = apply_window(ax, ApplyWindowInput(signal=ones, window="hann"))
    assert result.error.code == ""

    expected = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
    assert np.allclose(np.array(result.signal.values), expected, atol=1e-9)
    assert result.signal.values[0] == 0.0
    assert abs(result.signal.values[n // 2] - 1.0) < 0.01


def test_apply_window_rejects_unknown_window():
    ax = _TestContext()
    result = apply_window(ax, ApplyWindowInput(signal=Signal(values=[1.0, 2.0, 3.0], sample_rate_hz=10.0), window="triangle-of-mystery"))
    assert result.error.code == "INVALID_ARGUMENT"
