"""Envelope (demodulation) analysis for MCSA.

Hilbert‑transform‑based amplitude demodulation to extract low‑frequency
modulation patterns caused by mechanical faults (bearing defects,
eccentricity, load oscillations).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import signal as sig


def hilbert_envelope(
    x: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute the amplitude envelope of a signal via the Hilbert transform.

    Args:
        x: Input signal (real‑valued).

    Returns:
        Instantaneous amplitude (envelope) array.
    """
    analytic = sig.hilbert(x)
    return np.abs(analytic)


def instantaneous_frequency(
    x: NDArray[np.floating],
    fs: float,
) -> NDArray[np.floating]:
    """Compute instantaneous frequency via the Hilbert transform.

    Args:
        x: Input signal.
        fs: Sampling frequency in Hz.

    Returns:
        Instantaneous frequency array in Hz (length = len(x) - 1).
    """
    analytic = sig.hilbert(x)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase) / (2.0 * np.pi) * fs
    return inst_freq


def envelope_spectrum(
    x: NDArray[np.floating],
    fs: float,
    bandpass: tuple[float, float] | None = None,
    filter_order: int = 5,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute the spectrum of the signal envelope.

    Typical workflow for bearing / mechanical fault detection:
      1. Optional bandpass to isolate a resonance band
      2. Hilbert envelope
      3. Remove DC of envelope
      4. FFT of envelope → low‑frequency modulation spectrum

    Args:
        x: Input current signal.
        fs: Sampling frequency in Hz.
        bandpass: Optional (low, high) bandpass range in Hz before
                  computing the envelope (e.g. to isolate a resonance).
        filter_order: Butterworth filter order for bandpass.

    Returns:
        (frequencies, amplitudes) of the envelope spectrum.
    """
    y = x.copy()

    # Optional bandpass
    if bandpass is not None:
        nyq = fs / 2.0
        low, high = bandpass
        if 0 < low < high < nyq:
            sos = sig.butter(filter_order, [low / nyq, high / nyq], btype="bandpass", output="sos")
            y = sig.sosfiltfilt(sos, y)

    # Envelope
    env = hilbert_envelope(y)

    # Remove DC from envelope
    env = env - np.mean(env)

    # FFT of envelope
    n = len(env)
    n // 2 + 1
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    X = np.fft.rfft(env)
    amps = (2.0 / n) * np.abs(X)
    amps[0] /= 2.0

    return freqs, amps
