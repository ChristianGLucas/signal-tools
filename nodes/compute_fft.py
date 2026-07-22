import numpy as np
from scipy import fft as spfft

from gen.messages_pb2 import ComputeFFTInput, SpectrumResult
from gen.axiom_context import AxiomContext
from nodes._common import err, validate_signal, to_pylist


def compute_fft(ax: AxiomContext, input: ComputeFFTInput) -> SpectrumResult:
    """Computes the discrete Fourier transform of a real-valued signal via
    scipy.fft.rfft, returning the non-negative-frequency magnitude and phase
    spectrum with each bin's frequency in Hz. Requires sample_rate_hz > 0 on
    the input signal (needed to compute the frequency bins).
    """
    values, rate, e = validate_signal(input.signal, min_len=2, require_rate=True)
    if e:
        return SpectrumResult(error=e)

    n = input.n if input.n > 0 else len(values)
    if n > 40_000:
        return SpectrumResult(error=err("LIMIT_EXCEEDED", f"n must be <= 40000, got {n}"))

    spectrum = spfft.rfft(values, n=n)
    freqs = spfft.rfftfreq(n, d=1.0 / rate)
    return SpectrumResult(
        frequencies_hz=to_pylist(freqs),
        magnitude=to_pylist(np.abs(spectrum)),
        phase_rad=to_pylist(np.angle(spectrum)),
    )
