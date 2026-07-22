from gen.messages_pb2 import CoherenceInput, Signal
from nodes.compute_coherence import compute_coherence
from nodes._test_helpers import _TestContext, two_tone_signal


def test_compute_coherence_of_a_signal_with_itself_is_1():
    """Oracle: the magnitude-squared coherence of any signal with an exact
    copy of itself is 1.0 at every frequency (Cxy = Pxx, so
    |Cxy|^2/(Pxx*Pyy) = 1) — a mathematical identity independent of this
    node's code.
    """
    ax = _TestContext()
    sig = two_tone_signal()
    result = compute_coherence(ax, CoherenceInput(signal_a=sig, signal_b=sig, nperseg=256))
    assert result.error.code == ""
    assert all(c > 0.99 for c in result.coherence)


def test_compute_coherence_rejects_length_mismatch():
    ax = _TestContext()
    result = compute_coherence(ax, CoherenceInput(
        signal_a=Signal(values=[1.0, 2.0, 3.0, 4.0], sample_rate_hz=10.0),
        signal_b=Signal(values=[1.0, 2.0], sample_rate_hz=10.0),
    ))
    assert result.error.code == "INVALID_INPUT"
