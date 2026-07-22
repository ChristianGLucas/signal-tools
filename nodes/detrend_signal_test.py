import numpy as np

from gen.messages_pb2 import DetrendInput, Signal
from nodes.detrend_signal import detrend_signal
from nodes._test_helpers import _TestContext


def test_detrend_signal_linear_zeroes_out_a_perfect_ramp():
    """Oracle: a perfectly linear ramp has, by definition, zero residual
    after best-fit-line removal — a direct arithmetic fact, not something
    that depends on this node's implementation being "close enough."
    """
    ax = _TestContext()
    ramp = [float(i) for i in range(20)]
    result = detrend_signal(ax, DetrendInput(signal=Signal(values=ramp, sample_rate_hz=1.0), detrend_type="linear"))
    assert result.error.code == ""
    assert np.allclose(result.signal.values, 0.0, atol=1e-9)


def test_detrend_signal_constant_removes_the_mean():
    ax = _TestContext()
    values = [5.0, 7.0, 3.0, 9.0, 1.0]  # mean = 5.0
    result = detrend_signal(ax, DetrendInput(signal=Signal(values=values, sample_rate_hz=1.0), detrend_type="constant"))
    assert result.error.code == ""
    assert abs(sum(result.signal.values) / len(result.signal.values)) < 1e-9
    assert np.allclose(result.signal.values, [0.0, 2.0, -2.0, 4.0, -4.0], atol=1e-9)


def test_detrend_signal_rejects_unknown_type():
    ax = _TestContext()
    result = detrend_signal(ax, DetrendInput(signal=Signal(values=[1.0, 2.0, 3.0], sample_rate_hz=1.0), detrend_type="quadratic"))
    assert result.error.code == "INVALID_ARGUMENT"
