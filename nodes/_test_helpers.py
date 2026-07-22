"""Shared test scaffolding + oracle fixtures for christiangeorgelucas/signal-tools.

The two-tone fixture is chosen so its DFT has an EXACT, hand-computable
closed form: fs=1000 Hz, N=2000 samples (2s) means f1=50Hz completes exactly
100 cycles and f2=120Hz completes exactly 240 cycles in the window, so there
is zero spectral leakage and the rfft magnitude at bin k for a component
A*sin(2*pi*k*n/N) is exactly A*N/2 (a well-known closed-form DFT identity,
independent of this package's implementation) — this is the "signal with
known spectral content" oracle the package's tests are built around.
"""
from __future__ import annotations

import numpy as np

from gen.messages_pb2 import Signal
from gen.axiom_context import SecretStatus


class _TestContext:
    """Minimal AxiomContext implementation for unit tests."""

    class _Logger:
        def debug(self, msg: str, **attrs) -> None: pass
        def info(self, msg: str, **attrs) -> None: pass
        def warn(self, msg: str, **attrs) -> None: pass
        def error(self, msg: str, **attrs) -> None: pass

    class _Secrets:
        def __init__(self, m: dict, revoked: set) -> None:
            self._m = m or {}
            self._revoked = revoked or set()
        def get(self, name: str):
            v = self._m.get(name)
            return (v, True) if v is not None else ("", False)
        def status(self, name: str) -> SecretStatus:
            if name in self._m:
                return SecretStatus.AVAILABLE
            if name in self._revoked:
                return SecretStatus.REVOKED
            return SecretStatus.UNSET

    def __init__(self, secrets_map: dict | None = None, revoked_names: set | None = None) -> None:
        self.log = self._Logger()
        self.secrets = self._Secrets(secrets_map or {}, revoked_names)
        self.execution_id = "test-execution-id"
        self.flow_id = "test-flow-id"
        self.tenant_id = "test-tenant-id"


# Fixture parameters, shared by every test that needs a "known spectrum" signal.
FS = 1000.0
N = 2000
F1, A1 = 50.0, 1.0
F2, A2 = 120.0, 0.5
BIN1 = int(round(F1 * N / FS))   # == 100, exact
BIN2 = int(round(F2 * N / FS))   # == 240, exact


def two_tone_values() -> np.ndarray:
    t = np.arange(N) / FS
    return A1 * np.sin(2 * np.pi * F1 * t) + A2 * np.sin(2 * np.pi * F2 * t)


def two_tone_signal() -> Signal:
    return Signal(values=two_tone_values().tolist(), sample_rate_hz=FS)


def make_signal(values, rate: float = FS) -> Signal:
    return Signal(values=[float(v) for v in values], sample_rate_hz=float(rate))
