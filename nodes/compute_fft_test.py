from gen.messages_pb2 import ComputeFFTInput, Signal
from nodes.compute_fft import compute_fft
from nodes._test_helpers import _TestContext, two_tone_signal, FS, N, BIN1, BIN2, A1, A2


def test_compute_fft_finds_both_known_tones_at_exact_bins():
    """Oracle: for x[n] = A*sin(2*pi*k*n/N) with k an exact integer bin
    index, the closed-form rfft magnitude at bin k is exactly A*N/2 (no
    spectral leakage since the fixture completes a whole number of cycles).
    This is a hand-derived DFT identity, independent of this node's code.
    """
    ax = _TestContext()
    result = compute_fft(ax, ComputeFFTInput(signal=two_tone_signal()))
    assert result.error.code == ""
    assert len(result.magnitude) == N // 2 + 1
    assert len(result.frequencies_hz) == len(result.magnitude)

    assert abs(result.frequencies_hz[BIN1] - 50.0) < 1e-6
    assert abs(result.frequencies_hz[BIN2] - 120.0) < 1e-6
    assert abs(result.magnitude[BIN1] - A1 * N / 2) < 1.0
    assert abs(result.magnitude[BIN2] - A2 * N / 2) < 1.0

    # every other bin should carry negligible energy
    other_bins_energy = sum(
        m for i, m in enumerate(result.magnitude) if i not in (0, BIN1, BIN2)
    )
    assert other_bins_energy < 1.0


def test_compute_fft_requires_sample_rate():
    ax = _TestContext()
    result = compute_fft(ax, ComputeFFTInput(signal=Signal(values=[1.0, 2.0, 3.0, 4.0], sample_rate_hz=0.0)))
    assert result.error.code == "INVALID_ARGUMENT"


def test_compute_fft_rejects_too_short_signal():
    ax = _TestContext()
    result = compute_fft(ax, ComputeFFTInput(signal=Signal(values=[1.0], sample_rate_hz=FS)))
    assert result.error.code == "INVALID_INPUT"
