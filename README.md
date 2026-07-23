# christiangeorgelucas/signal-tools

Composable [Axiom](https://axiomide.com) nodes for general-purpose digital signal
processing (DSP) on caller-supplied 1D numeric signals — IoT/sensor
telemetry, biosignals (ECG/EEG), vibration/mechanical data, or any other
time series. Not audio-specific (see `audio-tools` for audio-feature
extraction) and not statistical/econometric time-series analysis (see
`time-series-tools`).

Wraps [SciPy](https://scipy.org) (`scipy.signal` + `scipy.fft`) and NumPy,
both BSD-3-Clause.

## Nodes

- **DesignFilter** — design a Butterworth/Chebyshev-I/Chebyshev-II/Bessel/
  elliptic lowpass/highpass/bandpass/bandstop digital filter; returns
  coefficients (SOS or b/a) without applying them.
- **ApplyFilter** — apply filter coefficients to a signal (`lfilter` or
  zero-phase `filtfilt`).
- **FilterSignal** — design + apply in one call.
- **ComputeFFT** — real-signal FFT: magnitude/phase spectrum with frequency
  bins.
- **ComputePSD** — power spectral density (Welch's method or periodogram).
- **ComputeSpectrogram** — STFT spectrogram, downsampled to respect the
  transport cap.
- **FindPeaks** — local maxima with height/threshold/distance/prominence/
  width.
- **ComputeEnvelope** — Hilbert-transform analytic-signal envelope
  (amplitude, phase, instantaneous frequency).
- **Convolve** — convolve two signals.
- **CrossCorrelate** — cross-correlation / autocorrelation with optional
  normalization.
- **ResampleSignal** — FFT-based resampling to a target length.
- **DecimateSignal** — anti-aliased integer-factor downsampling.
- **ApplyWindow** — Hann/Hamming/Blackman/Kaiser windowing.
- **DetrendSignal** — remove a linear or constant trend.
- **FrequencyResponse** — a filter's magnitude/phase/group-delay response.
- **SmoothSignal** — Savitzky-Golay smoothing.
- **ComputeRMSEnvelope** — short-time RMS energy envelope.
- **EstimateDominantFrequency** — the FFT bin with peak magnitude.
- **ComputeCoherence** — magnitude-squared coherence between two signals.

Every node is a stateless, deterministic, single-input/single-output
transform with a consistent structured-error contract (`Error{code,
message}`); degenerate input (empty, too short, invalid cutoff) returns a
structured error rather than crashing. Signals are capped at 20,000 samples
per input to bound CPU cost and keep responses within the platform's ~4 MiB
transport cap.

Built for the Axiom marketplace (handle `christiangeorgelucas`).

## License

MIT — Copyright (c) 2026 Christian George Lucas.
