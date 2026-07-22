from scipy import signal as sps

from gen.messages_pb2 import CoherenceInput, CoherenceResult
from gen.axiom_context import AxiomContext
from nodes._common import err, validate_signal, to_pylist


def compute_coherence(ax: AxiomContext, input: CoherenceInput) -> CoherenceResult:
    """Computes the magnitude-squared coherence between two EQUAL-LENGTH
    signals via scipy.signal.coherence — a per-frequency measure from 0
    (unrelated) to 1 (perfectly linearly related) of how well one signal's
    frequency content predicts the other's. Uses signal_a's sample_rate_hz;
    if signal_b's differs (and is set), that is rejected as inconsistent.
    """
    a, rate, e = validate_signal(input.signal_a, min_len=2, require_rate=True, name="signal_a")
    if e:
        return CoherenceResult(error=e)
    b, rate_b, e = validate_signal(input.signal_b, min_len=2, name="signal_b")
    if e:
        return CoherenceResult(error=e)

    if len(a) != len(b):
        return CoherenceResult(error=err("INVALID_INPUT", f"signal_a and signal_b must have equal length, got {len(a)} and {len(b)}"))
    if rate_b > 0 and abs(rate_b - rate) > 1e-9:
        return CoherenceResult(error=err("INVALID_ARGUMENT", f"signal_a.sample_rate_hz ({rate}) and signal_b.sample_rate_hz ({rate_b}) must match"))

    nperseg = input.nperseg if input.nperseg > 0 else None
    if nperseg is not None and nperseg > len(a):
        return CoherenceResult(error=err("INVALID_ARGUMENT", f"nperseg ({nperseg}) must not exceed the signal length ({len(a)})"))

    try:
        freqs, coh = sps.coherence(a, b, fs=rate, nperseg=nperseg)
    except Exception as exc:
        return CoherenceResult(error=err("COMPUTE_ERROR", f"coherence computation failed: {exc}"))

    return CoherenceResult(frequencies_hz=to_pylist(freqs), coherence=to_pylist(coh))
