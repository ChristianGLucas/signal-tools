import numpy as np
from scipy import signal as sps

from gen.messages_pb2 import SignalOnlyInput, EnvelopeResult
from gen.axiom_context import AxiomContext
from nodes._common import validate_signal, to_pylist


def compute_envelope(ax: AxiomContext, input: SignalOnlyInput) -> EnvelopeResult:
    """Computes the analytic-signal envelope of a real-valued signal via the
    Hilbert transform (scipy.signal.hilbert): instantaneous amplitude and
    (unwrapped) phase at every sample, plus instantaneous frequency in Hz
    (length = input length - 1) when the input signal's sample_rate_hz > 0.
    """
    values, rate, e = validate_signal(input.signal, min_len=2)
    if e:
        return EnvelopeResult(error=e)

    analytic = sps.hilbert(values)
    amplitude = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))

    inst_freq = []
    if rate > 0:
        inst_freq = np.diff(phase) / (2.0 * np.pi) * rate

    return EnvelopeResult(
        amplitude=to_pylist(amplitude),
        instantaneous_phase_rad=to_pylist(phase),
        instantaneous_frequency_hz=to_pylist(inst_freq),
    )
