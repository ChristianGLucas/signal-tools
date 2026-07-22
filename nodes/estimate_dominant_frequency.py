import numpy as np
from scipy import fft as spfft

from gen.messages_pb2 import SignalOnlyInput, DominantFrequencyResult
from gen.axiom_context import AxiomContext
from nodes._common import validate_signal


def estimate_dominant_frequency(ax: AxiomContext, input: SignalOnlyInput) -> DominantFrequencyResult:
    """Estimates a signal's single dominant frequency as the FFT bin with
    peak magnitude, excluding the DC (0 Hz) bin, via scipy.fft.rfft.
    Requires sample_rate_hz > 0 on the input signal.
    """
    values, rate, e = validate_signal(input.signal, min_len=2, require_rate=True)
    if e:
        return DominantFrequencyResult(error=e)

    spectrum = spfft.rfft(values)
    freqs = spfft.rfftfreq(len(values), d=1.0 / rate)
    magnitude = np.abs(spectrum)

    if len(magnitude) < 2:
        return DominantFrequencyResult(frequency_hz=0.0, magnitude=float(magnitude[0]) if len(magnitude) else 0.0)

    idx = 1 + int(np.argmax(magnitude[1:]))  # exclude DC bin 0
    return DominantFrequencyResult(frequency_hz=float(freqs[idx]), magnitude=float(magnitude[idx]))
