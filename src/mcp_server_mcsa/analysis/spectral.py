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

# Safety cap on adaptive FFT length (issue #1): 4 M samples → roughly
# 16 MB for the float64 magnitude spectrum, plus the complex FFT buffer.
# Beyond this an over-eager `min_resolution_hz` request would allocate
# multi-GB arrays, so we refuse with a descriptive ValueError instead.
_MAX_ADAPTIVE_N_FFT: int = 1 << 22

# Safety factor for adaptive resolution: requested ``min_resolution_hz``
# means the resulting bin width is at most ``min_resolution_hz / 4`` —
# tight enough that any peak within the user's tolerance window falls
# into at least one bin without ambiguity.
_RESOLUTION_SAFETY_FACTOR: int = 4


def _next_pow2_at_least(n: int) -> int:
    """Smallest power of two that is >= n. ``n`` must be positive."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def compute_fft_spectrum(
    x: NDArray[np.floating],
    fs: float,
    n_fft: int | None = None,
    sided: Literal["one", "two"] = "one",
    min_resolution_hz: float | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute the amplitude spectrum via FFT.

    Args:
        x: Input time‑domain signal (real‑valued).
        fs: Sampling frequency in Hz.
        n_fft: FFT length (zero‑padded). Default → len(x).
        sided: ``"one"`` for single‑sided (positive freqs only),
               ``"two"`` for full two‑sided spectrum.
        min_resolution_hz: When set, choose ``n_fft`` adaptively so the
            resulting bin width is at most ``min_resolution_hz / 4``.
            Downstream callers that want to resolve sidebands within a
            given tolerance should pass that tolerance here; the
            factor-4 safety margin guarantees the peak falls into at
            least one bin instead of between two. When both ``n_fft``
            and ``min_resolution_hz`` are passed, the larger of the
            two-derived values is used. Raises ``ValueError`` for
            non-positive values or requests that would exceed the
            internal safety cap (~16 MB spectrum).

    Returns:
        (frequencies, amplitudes) — both 1‑D arrays.

    Raises:
        ValueError: ``min_resolution_hz`` is non-positive, or the
            required ``n_fft`` exceeds the internal safety cap.
    """
    if min_resolution_hz is not None:
        if min_resolution_hz <= 0.0:
            raise ValueError(
                f"min_resolution_hz must be > 0, got {min_resolution_hz!r}"
            )
        required = int(np.ceil(fs * _RESOLUTION_SAFETY_FACTOR / min_resolution_hz))
        if required > _MAX_ADAPTIVE_N_FFT:
            raise ValueError(
                f"min_resolution_hz={min_resolution_hz} requires "
                f"n_fft={required} which exceeds the safety cap "
                f"{_MAX_ADAPTIVE_N_FFT}. Use a coarser resolution or "
                f"pass n_fft explicitly."
            )
        n_from_resolution = _next_pow2_at_least(required)
        # If n_fft is also passed explicitly, honour the larger value so
        # the user never loses resolution they asked for.
        n_fft = max(n_fft or len(x), n_from_resolution)

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
    interpolate: bool = False,
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
        interpolate: When ``True`` (opt-in, new in v0.3.0) refine each
            peak's frequency and amplitude with sub-bin parabolic
            interpolation (Smith MoDFT Sec 9.3). The default is ``False``
            in v0.3.0 to preserve byte-identical backward compatibility
            with v0.2.2 — every existing call site, including the three
            in ``server.py`` (find_spectrum_peaks, run_full_diagnosis,
            diagnose_from_file), continues to return the same
            bin-centred values. Pass ``interpolate=True`` to opt in to
            the refined values. The default is expected to flip to
            ``True`` in v0.4.0 along with a documented migration note —
            callers that rely on exact bin-centred frequencies (e.g.
            ``peak["frequency_hz"] == 50.0``) should pin
            ``interpolate=False`` explicitly before then.

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
