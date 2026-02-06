"""Time–frequency analysis for MCSA.

Short‑Time Fourier Transform (STFT) for analysing non‑stationary conditions
(variable speed/load, start‑up transients).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import signal as sig


def compute_stft(
    x: NDArray[np.floating],
    fs: float,
    nperseg: int = 256,
    noverlap: int | None = None,
    window: str = "hann",
) -> dict:
    """Compute the Short‑Time Fourier Transform of a signal.

    Args:
        x: Input signal.
        fs: Sampling frequency in Hz.
        nperseg: Window length per segment (samples).
        noverlap: Overlap between segments. Default → nperseg // 2.
        window: Window function name.

    Returns:
        Dictionary with:
          - ``frequencies_hz``: 1‑D array of frequency bins.
          - ``times_s``: 1‑D array of time centres.
          - ``magnitude``: 2‑D array (freq × time) of STFT magnitude.
          - ``n_freq_bins``: number of frequency bins.
          - ``n_time_bins``: number of time frames.
    """
    if noverlap is None:
        noverlap = nperseg // 2

    freqs, times, Zxx = sig.stft(
        x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap,
    )

    magnitude = np.abs(Zxx)

    return {
        "frequencies_hz": freqs,
        "times_s": times,
        "magnitude": magnitude,
        "n_freq_bins": len(freqs),
        "n_time_bins": len(times),
    }


def compute_spectrogram(
    x: NDArray[np.floating],
    fs: float,
    nperseg: int = 256,
    noverlap: int | None = None,
    window: str = "hann",
) -> dict:
    """Compute the spectrogram (magnitude‑squared STFT) of a signal.

    Convenience wrapper around scipy.signal.spectrogram.

    Args:
        x: Input signal.
        fs: Sampling frequency in Hz.
        nperseg: Segment length.
        noverlap: Segment overlap. Default → nperseg // 2.
        window: Window function.

    Returns:
        Dictionary with ``frequencies_hz``, ``times_s``, ``power``
        (2‑D array of spectral power values).
    """
    if noverlap is None:
        noverlap = nperseg // 2

    freqs, times, Sxx = sig.spectrogram(
        x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap,
    )

    return {
        "frequencies_hz": freqs,
        "times_s": times,
        "power": Sxx,
        "n_freq_bins": len(freqs),
        "n_time_bins": len(times),
    }


def track_frequency_over_time(
    stft_result: dict,
    target_freq_hz: float,
    tolerance_hz: float = 2.0,
) -> dict:
    """Track the amplitude of a specific frequency component over time.

    Useful for monitoring how a fault signature evolves during a transient
    (e.g. start‑up, load change).

    Args:
        stft_result: Output from ``compute_stft``.
        target_freq_hz: The target frequency to track.
        tolerance_hz: Frequency tolerance for peak search.

    Returns:
        Dictionary with ``times_s`` and ``amplitude`` arrays.
    """
    freqs = stft_result["frequencies_hz"]
    magnitude = stft_result["magnitude"]

    mask = np.abs(freqs - target_freq_hz) <= tolerance_hz
    if not np.any(mask):
        return {
            "target_freq_hz": target_freq_hz,
            "times_s": stft_result["times_s"].tolist(),
            "amplitude": [0.0] * stft_result["n_time_bins"],
            "found": False,
        }

    # Max amplitude in the tolerance band at each time step
    amps = np.max(magnitude[mask, :], axis=0)

    return {
        "target_freq_hz": target_freq_hz,
        "times_s": stft_result["times_s"].tolist(),
        "amplitude": amps.tolist(),
        "found": True,
    }
