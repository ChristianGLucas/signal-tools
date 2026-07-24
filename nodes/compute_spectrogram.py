import numpy as np
from scipy import signal as sps

from gen.messages_pb2 import ComputeSpectrogramInput, SpectrogramResult
from gen.axiom_context import AxiomContext
from nodes._common import err, validate_signal, to_pylist


def compute_spectrogram(ax: AxiomContext, input: ComputeSpectrogramInput) -> SpectrogramResult:
    """Computes a short-time Fourier transform spectrogram of a signal
    (scipy.signal.spectrogram, magnitude mode), summarized/downsampled
    (time-averaged) along the time axis to at most max_time_bins (default
    256) regardless of the input signal's length. Requires sample_rate_hz > 0.
    """
    values, rate, e = validate_signal(input.signal, min_len=8, require_rate=True)
    if e:
        return SpectrogramResult(error=e)

    nperseg = input.nperseg if input.nperseg > 0 else None
    if nperseg is not None and nperseg > len(values):
        return SpectrogramResult(error=err("INVALID_ARGUMENT", f"nperseg ({nperseg}) must not exceed the signal length ({len(values)})"))
    noverlap = input.noverlap if input.noverlap > 0 else None
    max_time_bins = input.max_time_bins if input.max_time_bins > 0 else 256

    try:
        freqs, times, mag = sps.spectrogram(values, fs=rate, nperseg=nperseg, noverlap=noverlap, mode="magnitude")
    except Exception as exc:
        return SpectrogramResult(error=err("COMPUTE_ERROR", f"spectrogram computation failed: {exc}"))

    n_time = mag.shape[1]
    if n_time > max_time_bins:
        # Downsample along the time axis by averaging into max_time_bins groups.
        edges = np.array_split(np.arange(n_time), max_time_bins)
        mag = np.stack([mag[:, grp].mean(axis=1) for grp in edges if len(grp)], axis=1)
        times = np.array([times[grp].mean() for grp in edges if len(grp)])

    n_freq_bins, n_time_bins = mag.shape
    return SpectrogramResult(
        frequencies_hz=to_pylist(freqs),
        times_s=to_pylist(times),
        magnitude_flat=to_pylist(mag.flatten()),
        n_freq_bins=n_freq_bins,
        n_time_bins=n_time_bins,
    )
