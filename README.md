# christiangeorgelucas/signal-tools

Composable [Axiom](https://axiomide.com) nodes for general-purpose digital signal
processing (DSP) on caller-supplied 1D numeric signals — IoT/sensor
telemetry, biosignals (ECG/EEG), vibration/mechanical data, or any other
time series. Not audio-specific (see `audio-tools` for audio-feature
extraction) and not statistical/econometric time-series analysis (see
`time-series-tools`).

Wraps [SciPy](https://scipy.org) (`scipy.signal` + `scipy.fft`) and NumPy,
both BSD-3-Clause.

## Use it from your agent or app

Every node in this package is a **live, auto-scaling API endpoint** on the
[Axiom](https://axiomide.com) marketplace — call it from an AI agent or your own
code, with nothing to self-host.

**📦 See it on the marketplace:**
https://dev.axiomide.com/marketplace/christiangeorgelucas/signal-tools@0.1.1

**Hook it up to an AI agent (MCP).** Add Axiom's hosted MCP server to any MCP
client and every node becomes a typed tool your agent can call — search the
catalog, inspect a schema, and invoke it directly.

```bash
# Claude Code
claude mcp add --transport http axiom https://api.axiomide.com/mcp \
  --header "Authorization: Bearer $AXIOM_API_KEY"
```

Claude Desktop, Cursor, or any config-based client:

```json
{
  "mcpServers": {
    "axiom": {
      "type": "http",
      "url": "https://api.axiomide.com/mcp",
      "headers": { "Authorization": "Bearer YOUR_AXIOM_API_KEY" }
    }
  }
}
```

**Call it from the CLI.**

```bash
axiom invoke christiangeorgelucas/signal-tools/DesignFilter --input '{ ... }'
```

**Call it over HTTP.**

```bash
curl -X POST https://api.axiomide.com/invocations/v1/nodes/christiangeorgelucas/signal-tools/0.1.1/DesignFilter \
  -H "Authorization: Bearer $AXIOM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{ ... }'
```

> Input/output schema for each node is on the marketplace page above, or via
> `axiom inspect node christiangeorgelucas/signal-tools/DesignFilter`.

### Get started free

Install the CLI:

```bash
# macOS / Linux — Homebrew
brew install axiomide/tap/axiom

# macOS / Linux — install script
curl -fsSL https://raw.githubusercontent.com/AxiomIDE/axiom-releases/main/install.sh | sh
```

**Windows:** download the `windows/amd64` `.zip` from the
[releases page](https://github.com/AxiomIDE/axiom-releases/releases), unzip it,
and put `axiom.exe` on your `PATH`.

Then `axiom version` to verify, `axiom login` (GitHub or Google) to authenticate,
and create an API key under **Console → API Keys**. Docs and sign-up at
**[axiomide.com](https://axiomide.com)**.

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
