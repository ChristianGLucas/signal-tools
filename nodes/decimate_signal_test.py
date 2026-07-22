from gen.messages_pb2 import DecimateInput, Signal
from nodes.decimate_signal import decimate_signal
from nodes._test_helpers import _TestContext, two_tone_signal, FS, N


def test_decimate_signal_reduces_length_and_rate_by_the_factor():
    """Oracle: decimating by factor F must produce exactly ceil(N/F) samples
    (scipy.signal.decimate's documented output length) at sample_rate/F — a
    directly checkable arithmetic fact about the operation's contract.
    """
    ax = _TestContext()
    factor = 4
    result = decimate_signal(ax, DecimateInput(signal=two_tone_signal(), factor=factor))
    assert result.error.code == ""
    assert len(result.signal.values) == -(-N // factor)  # ceil division
    assert abs(result.signal.sample_rate_hz - FS / factor) < 1e-9


def test_decimate_signal_rejects_factor_below_2():
    ax = _TestContext()
    result = decimate_signal(ax, DecimateInput(signal=Signal(values=[1.0, 2.0, 3.0], sample_rate_hz=10.0), factor=1))
    assert result.error.code == "INVALID_ARGUMENT"
