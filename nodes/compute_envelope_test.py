import numpy as np

from gen.messages_pb2 import SignalOnlyInput, Signal
from nodes.compute_envelope import compute_envelope
from nodes._test_helpers import _TestContext


def test_compute_envelope_recovers_amplitude_and_frequency_of_a_pure_tone():
    """Oracle: the analytic signal of A*sin(2*pi*f*t) is, by the closed-form
    definition of the Hilbert transform, A*exp(i*(2*pi*f*t - pi/2)) — a
    CONSTANT envelope amplitude A and a constant instantaneous frequency f
    (away from edge-effect transients). This is a textbook identity, not
    something derivable from this node's own implementation.
    """
    fs, f, amplitude = 1000.0, 25.0, 3.0
    t = np.arange(1000) / fs
    values = amplitude * np.sin(2 * np.pi * f * t)

    ax = _TestContext()
    result = compute_envelope(ax, SignalOnlyInput(signal=Signal(values=values.tolist(), sample_rate_hz=fs)))
    assert result.error.code == ""

    # Trim edge transients (Hilbert transform edge effects).
    interior_amp = np.array(result.amplitude)[100:-100]
    interior_freq = np.array(result.instantaneous_frequency_hz)[100:-100]

    assert np.allclose(interior_amp, amplitude, atol=0.05)
    assert np.allclose(interior_freq, f, atol=0.5)


def test_compute_envelope_rejects_too_short_signal():
    ax = _TestContext()
    result = compute_envelope(ax, SignalOnlyInput(signal=Signal(values=[1.0], sample_rate_hz=1.0)))
    assert result.error.code == "INVALID_INPUT"
