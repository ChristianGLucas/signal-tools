from gen.messages_pb2 import SignalOnlyInput, Signal
from nodes.estimate_dominant_frequency import estimate_dominant_frequency
from nodes._test_helpers import _TestContext, two_tone_signal, F1, A1, N


def test_estimate_dominant_frequency_finds_the_known_stronger_tone():
    """Oracle: in the two-tone fixture, a1=1.0 > a2=0.5, so the dominant
    frequency must be exactly the known 50Hz bin.
    """
    ax = _TestContext()
    result = estimate_dominant_frequency(ax, SignalOnlyInput(signal=two_tone_signal()))
    assert result.error.code == ""
    assert abs(result.frequency_hz - F1) < 1e-6
    assert abs(result.magnitude - A1 * N / 2) < 1.0


def test_estimate_dominant_frequency_requires_sample_rate():
    ax = _TestContext()
    result = estimate_dominant_frequency(ax, SignalOnlyInput(signal=Signal(values=[1.0, 2.0, 3.0], sample_rate_hz=0.0)))
    assert result.error.code == "INVALID_ARGUMENT"
