"""Spectral analysis utilities for MCSA.

FFT‑based spectrum, Welch PSD, and spectral peak detection.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import signal as sig

# Floor for log-magnitude interpolation (issue #3): prevents log10(0) when a
# peak's neighbour bin amplitude is exactly zero. Well below any meaningful
# physical amplitude so it never influences a real peak's refined value.
_LOG_AMP_FLOOR: float = 1e-12


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


def _parabolic_refine_peak(
    amps: NDArray[np.floating],
    idx: int,
    bin_width_hz: float,
) -> tuple[float, float]:
    """Sub-bin parabolic interpolation around the peak at ``amps[idx]``.

    Implements the standard quadratic interpolation in the log-magnitude
    domain (Smith, *Mathematics of the DFT*, Sec 9.3). For a windowed
    sinusoid the main lobe of the FFT magnitude is approximately
    parabolic in log scale, so fitting a parabola through three samples
    around the bin maximum recovers the true peak frequency and
    amplitude to within a few percent of the bin width.

    Args:
        amps: Linear-amplitude spectrum (the full ``amps`` array, not a
            slice — ``idx`` indexes into this directly).
        idx: Index of the bin with the local maximum.
        bin_width_hz: Frequency spacing between adjacent bins (Hz).

    Returns:
        ``(delta_hz, refined_amp_linear)`` where ``delta_hz`` is the
        sub-bin frequency offset (in (-bin_width/2, +bin_width/2)) the
        caller adds to ``freqs[idx]``, and ``refined_amp_linear`` is the
        interpolated peak amplitude in linear units.

    Edge cases (return ``(0.0, amps[idx])`` — no refinement):
        * ``idx`` at the array boundary (no neighbour on one side).
        * All three log-amplitudes equal (parabola is degenerate).
        * Bin width ≤ 0 (caller did not provide spacing).
    """
    n = len(amps)
    if idx <= 0 or idx >= n - 1 or bin_width_hz <= 0.0:
        return 0.0, float(amps[idx])

    # Log-magnitude interpolation; floor prevents log10(0).
    y_prev = float(np.log10(max(float(amps[idx - 1]), _LOG_AMP_FLOOR)))
    y_curr = float(np.log10(max(float(amps[idx]), _LOG_AMP_FLOOR)))
    y_next = float(np.log10(max(float(amps[idx + 1]), _LOG_AMP_FLOOR)))

    denom = y_prev - 2.0 * y_curr + y_next
    if denom == 0.0:
        return 0.0, float(amps[idx])

    delta_bins = 0.5 * (y_prev - y_next) / denom
    # Clamp to (-0.5, +0.5) — Smith's formula guarantees this for a true
    # local maximum, but numerical noise can occasionally push it slightly
    # outside on near-flat regions. The clamp keeps the result physically
    # meaningful (sub-bin offset within the central bin).
    if delta_bins > 0.5:
        delta_bins = 0.5
    elif delta_bins < -0.5:
        delta_bins = -0.5

    refined_log = y_curr - 0.25 * (y_prev - y_next) * delta_bins
    refined_amp = float(10.0**refined_log)
    delta_hz = delta_bins * bin_width_hz
    return delta_hz, refined_amp


def detect_peaks(
    freqs: NDArray[np.floating],
    amps: NDArray[np.floating],
    height: float | None = None,
    prominence: float | None = None,
    distance_hz: float | None = None,
    freq_range: tuple[float, float] | None = None,
    max_peaks: int = 50,
    interpolate: bool = True,
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
        interpolate: When True (default, since v0.3.0) refine each peak's
            frequency and amplitude with sub-bin parabolic interpolation
            (Smith MoDFT Sec 9.3). When False, returns bin-centered peaks
            exactly as v0.2.2 did — pass ``interpolate=False`` for
            byte-identical backward compatibility with that release.

    Returns:
        List of dicts with ``frequency_hz``, ``amplitude``, ``prominence``.
    """
    # Restrict to frequency range
    if freq_range is not None:
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
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

    # Bin width for sub-bin interpolation. Assumes uniform spacing, which
    # is true for the FFT spectra this module produces.
    bin_width_hz = (
        float(freqs_sub[1] - freqs_sub[0]) if len(freqs_sub) > 1 else 0.0
    )

    # Build result list
    results = []
    for i, pi in enumerate(peak_idx):
        pi_int = int(pi)
        f_center = float(freqs_sub[pi_int])
        a_center = float(amps_sub[pi_int])
        if interpolate and bin_width_hz > 0.0:
            delta_hz, refined_amp = _parabolic_refine_peak(
                amps_sub, pi_int, bin_width_hz
            )
            f_peak = f_center + delta_hz
            a_peak = refined_amp
        else:
            f_peak = f_center
            a_peak = a_center
        entry: dict = {
            "frequency_hz": f_peak,
            "amplitude": a_peak,
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
