"""Signal pre‑processing for MCSA.

Offset removal, normalisation, windowing, and digital filtering of
stator‑current time‑domain signals prior to spectral analysis.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import signal as sig


def remove_dc_offset(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Remove the DC (mean) component from a signal."""
    return x - np.mean(x)


def normalize_signal(
    x: NDArray[np.floating],
    nominal_current: float | None = None,
) -> NDArray[np.floating]:
    """Normalise signal amplitude.

    Args:
        x: Input signal.
        nominal_current: If provided, normalise to this value (p.u.).
            Otherwise normalise by RMS.

    Returns:
        Normalised signal array.
    """
    if nominal_current is not None and nominal_current > 0:
        return x / nominal_current
    rms = np.sqrt(np.mean(x ** 2))
    if rms == 0:
        return x
    return x / rms


def apply_window(
    x: NDArray[np.floating],
    window: Literal["hann", "hamming", "blackman", "flattop", "rectangular"] = "hann",
) -> NDArray[np.floating]:
    """Apply a window function to the signal.

    Args:
        x: Input signal array.
        window: Window type name.

    Returns:
        Windowed signal.
    """
    n = len(x)
    if window == "rectangular":
        return x.copy()
    w = sig.get_window(window, n)
    return x * w  # type: ignore[return-value]


def bandpass_filter(
    x: NDArray[np.floating],
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = 5,
) -> NDArray[np.floating]:
    """Apply a Butterworth bandpass filter.

    Args:
        x: Input signal.
        fs: Sampling frequency in Hz.
        low_hz: Lower cutoff frequency.
        high_hz: Upper cutoff frequency.
        order: Filter order.

    Returns:
        Bandpass‑filtered signal.
    """
    nyq = fs / 2.0
    if low_hz <= 0 or high_hz >= nyq or low_hz >= high_hz:
        raise ValueError(
            f"Invalid cutoff frequencies: low={low_hz}, high={high_hz}, Nyquist={nyq}"
        )
    sos = sig.butter(order, [low_hz / nyq, high_hz / nyq], btype="bandpass", output="sos")
    return sig.sosfiltfilt(sos, x)


def notch_filter(
    x: NDArray[np.floating],
    fs: float,
    notch_freq_hz: float,
    quality_factor: float = 30.0,
) -> NDArray[np.floating]:
    """Apply a notch (band‑reject) filter at a specific frequency.

    Useful for removing strong supply harmonics or converter switching noise.

    Args:
        x: Input signal.
        fs: Sampling frequency in Hz.
        notch_freq_hz: Centre frequency to reject.
        quality_factor: Q factor (higher = narrower notch).

    Returns:
        Filtered signal.
    """
    b, a = sig.iirnotch(notch_freq_hz, quality_factor, fs)
    return sig.filtfilt(b, a, x)


def lowpass_filter(
    x: NDArray[np.floating],
    fs: float,
    cutoff_hz: float,
    order: int = 5,
) -> NDArray[np.floating]:
    """Apply a Butterworth lowpass filter (anti‑aliasing or smoothing).

    Args:
        x: Input signal.
        fs: Sampling frequency in Hz.
        cutoff_hz: Cutoff frequency in Hz.
        order: Filter order.

    Returns:
        Lowpass‑filtered signal.
    """
    nyq = fs / 2.0
    if cutoff_hz <= 0 or cutoff_hz >= nyq:
        raise ValueError(f"Cutoff must be in (0, {nyq}), got {cutoff_hz}")
    sos = sig.butter(order, cutoff_hz / nyq, btype="low", output="sos")
    return sig.sosfiltfilt(sos, x)


def preprocess_pipeline(
    x: NDArray[np.floating],
    fs: float,
    nominal_current: float | None = None,
    window: Literal["hann", "hamming", "blackman", "flattop", "rectangular"] = "hann",
    bandpass: tuple[float, float] | None = None,
    notch_freqs: list[float] | None = None,
    notch_q: float = 30.0,
) -> NDArray[np.floating]:
    """Full pre‑processing pipeline for a stator‑current signal.

    Steps (in order):
      1. DC offset removal
      2. Optional notch filtering (e.g. supply harmonics)
      3. Optional bandpass filtering
      4. Normalisation
      5. Windowing

    Args:
        x: Raw current signal.
        fs: Sampling frequency in Hz.
        nominal_current: Nominal current for normalisation (A). None → RMS.
        window: Window function name.
        bandpass: Optional (low, high) Hz tuple for bandpass filtering.
        notch_freqs: Optional list of frequencies to notch out.
        notch_q: Q factor for notch filters.

    Returns:
        Pre‑processed signal ready for spectral analysis.
    """
    y = remove_dc_offset(x)

    if notch_freqs:
        for nf in notch_freqs:
            y = notch_filter(y, fs, nf, quality_factor=notch_q)

    if bandpass is not None:
        y = bandpass_filter(y, fs, bandpass[0], bandpass[1])

    y = normalize_signal(y, nominal_current)
    y = apply_window(y, window)

    return y
