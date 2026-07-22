from scipy import signal as sps

from gen.messages_pb2 import FindPeaksInput, PeaksResult
from gen.axiom_context import AxiomContext
from nodes._common import err, validate_signal, to_pylist, to_pyintlist


def find_peaks(ax: AxiomContext, input: FindPeaksInput) -> PeaksResult:
    """Locates local maxima in a signal via scipy.signal.find_peaks, with
    optional minimum-height/threshold/distance/prominence/width filters.
    Always returns each detected peak's index, height (the raw signal value
    at that index), prominence, and width (at relative height 0.5) —
    prominence and width are computed for every peak regardless of whether a
    minimum filter is supplied.
    """
    values, rate, e = validate_signal(input.signal, min_len=3)
    if e:
        return PeaksResult(error=e)

    distance = input.distance if input.HasField("distance") else None
    if distance is not None and distance < 1:
        return PeaksResult(error=err("INVALID_ARGUMENT", f"distance must be >= 1, got {distance}"))

    height = input.height if input.HasField("height") else None
    threshold = input.threshold if input.HasField("threshold") else None
    prominence_min = input.prominence if input.HasField("prominence") else 0.0
    width_min = input.width if input.HasField("width") else 0.0

    try:
        indices, props = sps.find_peaks(
            values, height=height, threshold=threshold, distance=distance,
            prominence=prominence_min, width=width_min,
        )
    except Exception as exc:
        return PeaksResult(error=err("COMPUTE_ERROR", f"peak detection failed: {exc}"))

    return PeaksResult(
        indices=to_pyintlist(indices),
        heights=to_pylist(values[indices]),
        prominences=to_pylist(props["prominences"]),
        widths=to_pylist(props["widths"]),
    )
