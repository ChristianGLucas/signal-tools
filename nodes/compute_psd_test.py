from gen.messages_pb2 import ComputePSDInput, Signal
from nodes.compute_psd import compute_psd
from nodes._test_helpers import _TestContext, two_tone_signal, FS


def test_compute_psd_welch_peaks_near_the_dominant_known_tone():
    """Oracle: the two-tone fixture's stronger component is at 50Hz (a1=1.0
    vs a2=0.5) — Welch's method should place its global PSD peak within a
    couple of frequency bins of 50Hz.
    """
    ax = _TestContext()
    result = compute_psd(ax, ComputePSDInput(signal=two_tone_signal(), method="welch"))
    assert result.error.code == ""
    assert len(result.frequencies_hz) == len(result.psd)

    peak_idx = max(range(len(result.psd)), key=lambda i: result.psd[i])
    peak_freq = result.frequencies_hz[peak_idx]
    assert abs(peak_freq - 50.0) < 4.0


def test_compute_psd_periodogram_runs_and_is_nonnegative():
    ax = _TestContext()
    result = compute_psd(ax, ComputePSDInput(signal=two_tone_signal(), method="periodogram"))
    assert result.error.code == ""
    assert all(p >= 0.0 for p in result.psd)


def test_compute_psd_rejects_unknown_method():
    ax = _TestContext()
    result = compute_psd(ax, ComputePSDInput(signal=Signal(values=[1.0, 2.0, 3.0, 4.0], sample_rate_hz=FS), method="bogus"))
    assert result.error.code == "INVALID_ARGUMENT"
