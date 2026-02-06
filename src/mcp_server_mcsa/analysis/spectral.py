"""Spectral analysis utilities for MCSA.

FFT‑based spectrum, Welch PSD, and spectral peak detection.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import signal as sig


def compute_fft_spectrum(
    x: NDArray[np.floating],
    fs: float,
    n_fft: int | None = None,
    sided: Literal["one", "two"] = "one",
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute the amplitude spectrum via FFT.

    Args:
        x: Input time‑domain signal (real‑valued).
        fs: Sampling frequency in Hz.
        n_fft: FFT length (zero‑padded). Default → len(x).
        sided: ``"one"`` for single‑sided (positive freqs only),
               ``"two"`` for full two‑sided spectrum.

    Returns:
        (frequencies, amplitudes) — both 1‑D arrays.
    """
    n = n_fft or len(x)
    X = np.fft.fft(x, n=n)

    if sided == "one":
        n_pos = n // 2 + 1
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        amps = (2.0 / len(x)) * np.abs(X[:n_pos])
        amps[0] /= 2.0  # DC component not doubled
        return freqs, amps
    else:
        freqs = np.fft.fftfreq(n, d=1.0 / fs)
        amps = (1.0 / len(x)) * np.abs(X)
        return freqs, amps


def compute_psd(
    x: NDArray[np.floating],
    fs: float,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    scaling: Literal["density", "spectrum"] = "density",
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute Power Spectral Density using Welch's method.

    Args:
        x: Input signal.
        fs: Sampling frequency in Hz.
        nperseg: FFT segment length. Default → len(x) // 8 or 256.
        noverlap: Overlap between segments. Default → nperseg // 2.
        window: Window function name.
        scaling: ``"density"`` → V²/Hz, ``"spectrum"`` → V².

    Returns:
        (frequencies, psd_values) arrays.
    """
    if nperseg is None:
        nperseg = min(len(x), max(256, len(x) // 8))

    freqs, psd = sig.welch(
        x, fs=fs, window=window, nperseg=nperseg,
        noverlap=noverlap, scaling=scaling,
    )
    return freqs, psd


def detect_peaks(
    freqs: NDArray[np.floating],
    amps: NDArray[np.floating],
    height: float | None = None,
    prominence: float | None = None,
    distance_hz: float | None = None,
    freq_range: tuple[float, float] | None = None,
    max_peaks: int = 50,
) -> list[dict]:
    """Detect spectral peaks and return their properties.

    Args:
        freqs: Frequency axis (Hz).
        amps: Amplitude or PSD values.
        height: Minimum peak height.
        prominence: Minimum peak prominence.
        distance_hz: Minimum distance between peaks in Hz.
        freq_range: Optional (low, high) Hz range to search within.
        max_peaks: Maximum number of peaks to return (sorted by amplitude).

    Returns:
        List of dicts with ``frequency_hz``, ``amplitude``, ``prominence``.
    """
    # Restrict to frequency range
    if freq_range is not None:
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        int(np.argmax(mask))
        freqs_sub = freqs[mask]
        amps_sub = amps[mask]
    else:
        freqs_sub = freqs
        amps_sub = amps

    # Convert distance_hz to samples
    if distance_hz is not None and len(freqs_sub) > 1:
        df = float(freqs_sub[1] - freqs_sub[0])
        distance_samples = max(1, int(distance_hz / df))
    else:
        distance_samples = None

    peak_idx, properties = sig.find_peaks(
        amps_sub,
        height=height,
        prominence=prominence,
        distance=distance_samples,
    )

    # Build result list
    results = []
    for i, pi in enumerate(peak_idx):
        entry: dict = {
            "frequency_hz": float(freqs_sub[pi]),
            "amplitude": float(amps_sub[pi]),
        }
        if "prominences" in properties:
            entry["prominence"] = float(properties["prominences"][i])
        results.append(entry)

    # Sort by amplitude descending, limit
    results.sort(key=lambda p: p["amplitude"], reverse=True)
    return results[:max_peaks]


def amplitude_at_frequency(
    freqs: NDArray[np.floating],
    amps: NDArray[np.floating],
    target_freq_hz: float,
    tolerance_hz: float = 0.5,
) -> dict:
    """Find the amplitude at (or nearest to) a target frequency.

    Args:
        freqs: Frequency axis.
        amps: Amplitude values.
        target_freq_hz: Frequency of interest.
        tolerance_hz: Search tolerance around target.

    Returns:
        Dict with ``frequency_hz``, ``amplitude``, ``found`` flag.
    """
    mask = np.abs(freqs - target_freq_hz) <= tolerance_hz
    if not np.any(mask):
        return {"frequency_hz": target_freq_hz, "amplitude": 0.0, "found": False}

    subset_amps = amps[mask]
    subset_freqs = freqs[mask]
    best = int(np.argmax(subset_amps))

    return {
        "frequency_hz": float(subset_freqs[best]),
        "amplitude": float(subset_amps[best]),
        "found": True,
    }
