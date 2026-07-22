import numpy as np

from gen.messages_pb2 import ResampleInput, Signal
from nodes.resample_signal import resample_signal
from nodes._test_helpers import _TestContext, two_tone_signal, FS, N, F1


def test_resample_signal_preserves_the_dominant_known_frequency():
    """Oracle: resampling to double length and re-analyzing with a plain
    numpy.fft.rfft (independent of this package's ComputeFFT node) should
    still show the fixture's stronger 50Hz component as the dominant
    spectral peak.
    """
    ax = _TestContext()
    target = N * 2
    result = resample_signal(ax, ResampleInput(signal=two_tone_signal(), target_length=target))
    assert result.error.code == ""
    assert len(result.signal.values) == target
    assert abs(result.signal.sample_rate_hz - FS * 2) < 1e-6

    mag = np.abs(np.fft.rfft(np.asarray(result.signal.values)))
    freqs = np.fft.rfftfreq(target, d=1.0 / result.signal.sample_rate_hz)
    peak_freq = freqs[int(np.argmax(mag[1:])) + 1]
    assert abs(peak_freq - F1) < 1.0


def test_resample_signal_rejects_target_length_out_of_bounds():
    ax = _TestContext()
    result = resample_signal(ax, ResampleInput(signal=Signal(values=[1.0, 2.0, 3.0], sample_rate_hz=10.0), target_length=0))
    assert result.error.code == "INVALID_ARGUMENT"
