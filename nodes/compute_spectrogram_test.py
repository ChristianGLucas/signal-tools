from gen.messages_pb2 import ComputeSpectrogramInput, Signal
from nodes.compute_spectrogram import compute_spectrogram
from nodes._test_helpers import _TestContext, two_tone_signal, FS


def test_compute_spectrogram_shows_more_energy_near_the_known_tone():
    """Oracle: a frequency bin near the fixture's known 50Hz component must
    carry substantially more average magnitude across time than a bin far
    from either known component (e.g. 400Hz, present in neither tone).
    """
    ax = _TestContext()
    result = compute_spectrogram(ax, ComputeSpectrogramInput(signal=two_tone_signal(), max_time_bins=32))
    assert result.error.code == ""
    assert result.n_freq_bins * result.n_time_bins == len(result.magnitude_flat)
    assert result.n_time_bins <= 32

    freqs = list(result.frequencies_hz)
    idx_50 = min(range(len(freqs)), key=lambda i: abs(freqs[i] - 50.0))
    idx_400 = min(range(len(freqs)), key=lambda i: abs(freqs[i] - 400.0))

    def row_mean(freq_idx):
        row = result.magnitude_flat[freq_idx * result.n_time_bins:(freq_idx + 1) * result.n_time_bins]
        return sum(row) / len(row)

    assert row_mean(idx_50) > 5 * row_mean(idx_400)


def test_compute_spectrogram_rejects_nperseg_larger_than_signal():
    ax = _TestContext()
    result = compute_spectrogram(ax, ComputeSpectrogramInput(
        signal=Signal(values=[float(i) for i in range(16)], sample_rate_hz=FS), nperseg=1000,
    ))
    assert result.error.code == "INVALID_ARGUMENT"
