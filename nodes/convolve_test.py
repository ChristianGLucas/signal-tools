from gen.messages_pb2 import TwoSignalsInput, Signal
from nodes.convolve import convolve
from nodes._test_helpers import _TestContext


def test_convolve_matches_hand_computed_full_convolution():
    """Oracle: convolving [1,1,1] with [1,1,1] by hand gives
    [1,2,3,2,1] — a direct arithmetic computation, independent of scipy.
    """
    ax = _TestContext()
    a = Signal(values=[1.0, 1.0, 1.0], sample_rate_hz=10.0)
    b = Signal(values=[1.0, 1.0, 1.0], sample_rate_hz=10.0)
    result = convolve(ax, TwoSignalsInput(signal_a=a, signal_b=b, convolve_mode="full"))
    assert result.error.code == ""
    assert [round(v, 6) for v in result.signal.values] == [1.0, 2.0, 3.0, 2.0, 1.0]
    assert result.signal.sample_rate_hz == 10.0


def test_convolve_same_mode_matches_expected_length():
    ax = _TestContext()
    a = Signal(values=[1.0, 2.0, 3.0, 4.0, 5.0], sample_rate_hz=10.0)
    b = Signal(values=[1.0, 0.0, -1.0], sample_rate_hz=10.0)
    result = convolve(ax, TwoSignalsInput(signal_a=a, signal_b=b, convolve_mode="same"))
    assert result.error.code == ""
    assert len(result.signal.values) == 5


def test_convolve_rejects_empty_signal():
    ax = _TestContext()
    result = convolve(ax, TwoSignalsInput(
        signal_a=Signal(values=[], sample_rate_hz=10.0),
        signal_b=Signal(values=[1.0], sample_rate_hz=10.0),
    ))
    assert result.error.code == "INVALID_INPUT"
