from gen.messages_pb2 import FindPeaksInput, Signal
from nodes.find_peaks import find_peaks
from nodes._test_helpers import _TestContext


# Hand-constructed signal: zeros with local maxima of known height at known
# indices — the oracle is direct visual/arithmetic inspection, independent
# of scipy: peaks at indices 1,3,5,7,9 with heights 1,2,3,2,1.
KNOWN_PEAKS_SIGNAL = [0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 2.0, 0.0, 1.0, 0.0]


def test_find_peaks_locates_all_known_peaks():
    ax = _TestContext()
    result = find_peaks(ax, FindPeaksInput(signal=Signal(values=KNOWN_PEAKS_SIGNAL, sample_rate_hz=1.0)))
    assert result.error.code == ""
    assert list(result.indices) == [1, 3, 5, 7, 9]
    assert list(result.heights) == [1.0, 2.0, 3.0, 2.0, 1.0]
    assert len(result.prominences) == 5
    assert len(result.widths) == 5


def test_find_peaks_height_filter_keeps_only_the_tallest_known_peak():
    ax = _TestContext()
    result = find_peaks(ax, FindPeaksInput(
        signal=Signal(values=KNOWN_PEAKS_SIGNAL, sample_rate_hz=1.0), height=2.5,
    ))
    assert result.error.code == ""
    assert list(result.indices) == [5]
    assert list(result.heights) == [3.0]


def test_find_peaks_rejects_nonpositive_distance():
    ax = _TestContext()
    result = find_peaks(ax, FindPeaksInput(
        signal=Signal(values=KNOWN_PEAKS_SIGNAL, sample_rate_hz=1.0), distance=0.0,
    ))
    assert result.error.code == "INVALID_ARGUMENT"
