import numpy as np

from gen.messages_pb2 import RMSEnvelopeInput, Signal
from nodes.compute_rms_envelope import compute_rms_envelope
from nodes._test_helpers import _TestContext


def test_compute_rms_envelope_of_constant_signal_equals_its_absolute_value():
    """Oracle: the RMS of a constant signal c is |c| by definition
    (sqrt(mean(c^2)) = sqrt(c^2) = |c|) — true for any window size.
    """
    ax = _TestContext()
    values = [-4.0] * 40
    result = compute_rms_envelope(ax, RMSEnvelopeInput(signal=Signal(values=values, sample_rate_hz=100.0), window_size=10))
    assert result.error.code == ""
    assert np.allclose(result.signal.values, 4.0, atol=1e-9)
    assert abs(result.signal.sample_rate_hz - 10.0) < 1e-9  # 100 / hop(=window=10)


def test_compute_rms_envelope_of_sine_matches_closed_form_over_full_periods():
    """Oracle: the RMS of A*sin over a whole number of periods is A/sqrt(2)
    — a standard closed-form AC-signal identity.
    """
    fs, f, amplitude = 1000.0, 10.0, 2.0
    n_periods = 20
    n = int(n_periods * fs / f)
    t = np.arange(n) / fs
    values = amplitude * np.sin(2 * np.pi * f * t)

    ax = _TestContext()
    result = compute_rms_envelope(ax, RMSEnvelopeInput(
        signal=Signal(values=values.tolist(), sample_rate_hz=fs), window_size=n,
    ))
    assert result.error.code == ""
    assert abs(result.signal.values[0] - amplitude / np.sqrt(2)) < 0.01


def test_compute_rms_envelope_rejects_window_larger_than_signal():
    ax = _TestContext()
    result = compute_rms_envelope(ax, RMSEnvelopeInput(signal=Signal(values=[1.0, 2.0, 3.0], sample_rate_hz=1.0), window_size=10))
    assert result.error.code == "INVALID_INPUT"
