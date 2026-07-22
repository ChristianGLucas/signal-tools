import numpy as np

from gen.messages_pb2 import FilterSignalInput
from nodes.filter_signal import filter_signal
from nodes._test_helpers import _TestContext, two_tone_signal, N, BIN1, BIN2


def _rfft_mag(values):
    return np.abs(np.fft.rfft(np.asarray(values)))


def test_filter_signal_bandpass_isolates_the_high_tone():
    """Oracle: a bandpass filter straddling only the 120Hz component of the
    two-tone fixture should isolate it and suppress the 50Hz component —
    checked independently via numpy.fft.
    """
    ax = _TestContext()
    result = filter_signal(ax, FilterSignalInput(
        signal=two_tone_signal(), design="butterworth", band_type="bandpass",
        order=4, cutoff_hz=[90.0, 150.0], mode="filtfilt",
    ))
    assert result.error.code == ""

    mag = _rfft_mag(result.signal.values)
    assert mag[BIN2] > 300.0
    assert mag[BIN1] < 0.05 * mag[BIN2]


def test_filter_signal_requires_sample_rate():
    ax = _TestContext()
    from gen.messages_pb2 import Signal
    result = filter_signal(ax, FilterSignalInput(
        signal=Signal(values=[1.0, 2.0, 3.0], sample_rate_hz=0.0),
        band_type="lowpass", cutoff_hz=[10.0],
    ))
    assert result.error.code == "INVALID_ARGUMENT"
