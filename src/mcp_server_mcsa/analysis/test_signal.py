"""Synthetic test‑signal generator for MCSA.

Generates simulated stator‑current waveforms with optional fault
signatures for testing, validation, and demonstration purposes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def generate_healthy_signal(
    duration_s: float,
    fs_sample: float,
    supply_freq_hz: float = 50.0,
    amplitude: float = 1.0,
    noise_std: float = 0.01,
    harmonics: list[tuple[int, float]] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Generate a clean (healthy) motor‑current signal.

    Args:
        duration_s: Signal duration in seconds.
        fs_sample: Sampling frequency in Hz.
        supply_freq_hz: Supply fundamental frequency in Hz.
        amplitude: Peak amplitude of the fundamental.
        noise_std: Standard deviation of additive Gaussian noise.
        harmonics: Optional list of (harmonic_number, relative_amplitude)
            to add supply harmonics (e.g. [(3, 0.05), (5, 0.03)]).

    Returns:
        (time, signal) — 1‑D arrays.
    """
    n_samples = int(duration_s * fs_sample)
    t = np.arange(n_samples) / fs_sample

    x = amplitude * np.sin(2.0 * np.pi * supply_freq_hz * t)

    if harmonics:
        for h_num, h_amp in harmonics:
            x += h_amp * amplitude * np.sin(2.0 * np.pi * h_num * supply_freq_hz * t)

    if noise_std > 0:
        x += np.random.default_rng(42).normal(0, noise_std, n_samples)

    return t, x


def inject_brb_fault(
    t: NDArray[np.floating],
    x: NDArray[np.floating],
    supply_freq_hz: float,
    slip: float,
    sideband_amplitude: float = 0.02,
    n_harmonics: int = 1,
) -> NDArray[np.floating]:
    """Inject broken‑rotor‑bar sidebands into a current signal.

    Adds components at (1 ± 2ks)·f_s for k = 1 … n_harmonics.

    Args:
        t: Time axis.
        x: Original signal.
        supply_freq_hz: Supply frequency (Hz).
        slip: Motor slip (per‑unit).
        sideband_amplitude: Amplitude of each sideband component.
        n_harmonics: Number of sideband pairs.

    Returns:
        Signal with BRB fault injected.
    """
    y = x.copy()
    fs = supply_freq_hz

    for k in range(1, n_harmonics + 1):
        f_lower = (1 - 2 * k * slip) * fs
        f_upper = (1 + 2 * k * slip) * fs
        y += sideband_amplitude * np.sin(2.0 * np.pi * f_lower * t)
        y += sideband_amplitude * np.sin(2.0 * np.pi * f_upper * t)

    return y


def inject_eccentricity_fault(
    t: NDArray[np.floating],
    x: NDArray[np.floating],
    supply_freq_hz: float,
    rotor_freq_hz: float,
    sideband_amplitude: float = 0.015,
    n_harmonics: int = 2,
) -> NDArray[np.floating]:
    """Inject eccentricity sidebands at f_s ± k·f_r.

    Args:
        t: Time axis.
        x: Original signal.
        supply_freq_hz: Supply frequency (Hz).
        rotor_freq_hz: Rotor mechanical frequency (Hz).
        sideband_amplitude: Amplitude per sideband.
        n_harmonics: Number of sideband pairs.

    Returns:
        Signal with eccentricity fault injected.
    """
    y = x.copy()

    for k in range(1, n_harmonics + 1):
        f_lo = supply_freq_hz - k * rotor_freq_hz
        f_hi = supply_freq_hz + k * rotor_freq_hz
        y += sideband_amplitude * np.sin(2.0 * np.pi * f_lo * t)
        y += sideband_amplitude * np.sin(2.0 * np.pi * f_hi * t)

    return y


def inject_bearing_fault(
    t: NDArray[np.floating],
    x: NDArray[np.floating],
    supply_freq_hz: float,
    defect_freq_hz: float,
    modulation_depth: float = 0.01,
    n_harmonics: int = 2,
) -> NDArray[np.floating]:
    """Inject bearing defect modulation into a current signal.

    Bearing faults produce torque oscillations that amplitude‑modulate
    the stator current, creating sidebands at f_s ± k·f_defect.

    Args:
        t: Time axis.
        x: Original signal.
        supply_freq_hz: Supply frequency (Hz).
        defect_freq_hz: Bearing characteristic defect frequency (Hz).
        modulation_depth: Modulation amplitude.
        n_harmonics: Number of sideband pairs.

    Returns:
        Signal with bearing fault injected.
    """
    y = x.copy()

    for k in range(1, n_harmonics + 1):
        f_lo = supply_freq_hz - k * defect_freq_hz
        f_hi = supply_freq_hz + k * defect_freq_hz
        y += modulation_depth * np.sin(2.0 * np.pi * f_lo * t)
        y += modulation_depth * np.sin(2.0 * np.pi * f_hi * t)

    return y


def generate_test_signal(
    duration_s: float = 10.0,
    fs_sample: float = 5000.0,
    supply_freq_hz: float = 50.0,
    poles: int = 4,
    rotor_speed_rpm: float = 1470.0,
    amplitude: float = 1.0,
    noise_std: float = 0.01,
    faults: list[str] | None = None,
    fault_severity: float = 0.02,
    bearing_defect_freq_hz: float | None = None,
) -> dict:
    """Generate a complete synthetic motor‑current test signal.

    Convenience function that creates a healthy baseline and optionally
    injects one or more fault types.

    Args:
        duration_s: Signal duration in seconds.
        fs_sample: Sampling frequency in Hz.
        supply_freq_hz: Supply frequency in Hz.
        poles: Number of poles.
        rotor_speed_rpm: Rotor speed in RPM.
        amplitude: Fundamental amplitude.
        noise_std: Noise level.
        faults: List of fault types to inject. Options:
            ``"brb"`` — broken rotor bars,
            ``"eccentricity"`` — air‑gap eccentricity,
            ``"bearing"`` — bearing defect.
        fault_severity: Amplitude of fault components (0–1 relative).
        bearing_defect_freq_hz: Bearing defect frequency (Hz). Required
            if ``"bearing"`` is in faults.

    Returns:
        Dictionary with ``time_s``, ``signal``, ``sampling_freq_hz``,
        ``motor_params``, ``faults_injected``.
    """
    sync_speed = 120.0 * supply_freq_hz / poles
    slip = (sync_speed - rotor_speed_rpm) / sync_speed
    rotor_freq = rotor_speed_rpm / 60.0

    t, x = generate_healthy_signal(
        duration_s, fs_sample, supply_freq_hz, amplitude, noise_std,
        harmonics=[(3, 0.03), (5, 0.015)],
    )

    injected = []

    if faults:
        for fault in faults:
            fault = fault.lower().strip()
            if fault == "brb":
                x = inject_brb_fault(t, x, supply_freq_hz, slip, fault_severity)
                injected.append("broken_rotor_bars")
            elif fault == "eccentricity":
                x = inject_eccentricity_fault(
                    t, x, supply_freq_hz, rotor_freq, fault_severity
                )
                injected.append("eccentricity")
            elif fault == "bearing":
                bdf = bearing_defect_freq_hz or (3.5 * rotor_freq)
                x = inject_bearing_fault(t, x, supply_freq_hz, bdf, fault_severity)
                injected.append("bearing")

    return {
        "time_s": t.tolist(),
        "signal": x.tolist(),
        "sampling_freq_hz": fs_sample,
        "duration_s": duration_s,
        "n_samples": len(t),
        "motor_params": {
            "supply_freq_hz": supply_freq_hz,
            "poles": poles,
            "sync_speed_rpm": sync_speed,
            "rotor_speed_rpm": rotor_speed_rpm,
            "slip": slip,
            "rotor_freq_hz": rotor_freq,
        },
        "faults_injected": injected,
        "fault_severity": fault_severity,
    }
