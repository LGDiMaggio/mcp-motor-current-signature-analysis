"""Fault detection and severity assessment for MCSA.

Computes standardised fault indices from current spectra and provides
severity classification based on configurable thresholds.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mcp_server_mcsa.analysis.motor import MotorParameters
from mcp_server_mcsa.analysis.spectral import amplitude_at_frequency

# ---------------------------------------------------------------------------
# Severity thresholds (dB below fundamental)
# ---------------------------------------------------------------------------
# These are widely‑used empirical guidelines for induction motors.
# They should be adapted to the specific motor/application.
BRB_THRESHOLDS = {
    "healthy": -50.0,       # dB — sideband ≤ -50 dB relative to fundamental
    "incipient": -45.0,     # dB — early-stage fault
    "moderate": -40.0,      # dB — developing fault
    "severe": -35.0,        # dB — immediate action recommended
}

ECCENTRICITY_THRESHOLDS = {
    "healthy": -50.0,
    "incipient": -44.0,
    "moderate": -38.0,
    "severe": -30.0,
}


def _db_ratio(a: float, ref: float) -> float:
    """Compute 20·log10(a / ref), safe for zero values."""
    if ref <= 0 or a <= 0:
        return -np.inf
    return 20.0 * np.log10(a / ref)


# --- detection_status helper (issue #2) ---------------------------------
# Factor-4 safety margin: bin width must be at most tolerance_hz/4 for a
# peak inside the tolerance window to land into at least one bin reliably.
_RESOLUTION_SAFETY_FACTOR: float = 4.0

# Noise-floor threshold for the ``detection_status.detected`` flag.
# A bin within the tolerance window may carry a tiny numerical-leakage
# amplitude even on a clean synthetic signal — the resulting dB reading
# is finite but several orders of magnitude below the severity
# thresholds (which start at -50 dB). Any headline_db below this floor
# is treated as "not detected" so the ``no_sideband_present`` reason
# path can fire correctly on clean signals; downstream consumers
# wanting the raw value can still read ``worst_sideband_db`` /
# ``combined_index_db`` directly.
_DETECTION_NOISE_FLOOR_DB: float = -90.0


def _build_detection_status(
    *,
    freqs: NDArray[np.floating],
    tolerance_hz: float,
    headline_db: float,
    expected_sideband_freqs_hz: list[float],
    brb_sideband_distance_hz: float | None = None,
) -> dict:
    """Construct the ``detection_status`` block per issue #2.

    Args:
        freqs: Frequency axis of the spectrum being inspected.
        tolerance_hz: Tolerance window used by the caller for sideband
            matching (also the value reported in the status block).
        headline_db: The fault-index's single consolidated dB value
            (``worst_sideband_db`` for bearing / eccentricity /
            stator-interturn; ``combined_index_db`` for BRB). ``-inf``
            means no sideband contributed to it.
        expected_sideband_freqs_hz: All sideband locations the caller
            searched, including negative or super-Nyquist ones. Used to
            detect the ``frequency_out_of_range`` case.
        brb_sideband_distance_hz: Only the BRB caller sets this — the
            distance from the supply line to the slip sideband
            (``2·s·fs``). When the distance is smaller than the supply
            line's own main-lobe half-width (estimated from the bin
            width), reports the BRB-specific
            ``sideband_inside_supply_main_lobe`` reason.

    Returns:
        The five-field ``detection_status`` dict the spec describes.

    Reason priority (most fundamental first):
        1. ``detected`` — ``headline_db`` is finite.
        2. ``frequency_out_of_range`` — every expected sideband is
           outside the spectrum's physical range.
        3. ``sideband_inside_supply_main_lobe`` — BRB only; expected
           sideband sits inside the supply line's main lobe so no
           amount of zero-padding can resolve it (the time-domain
           window is too short).
        4. ``frequency_resolution_insufficient`` — bin width exceeds
           ``tolerance_hz / 4``; the search window doesn't contain a
           bin.
        5. ``no_sideband_present`` — none of the above explains the
           absence; the bearing/rotor is plausibly healthy.
    """
    if len(freqs) > 1:
        fft_bin_width_hz = float(freqs[1] - freqs[0])
    else:
        fft_bin_width_hz = 0.0
    min_bin_width = tolerance_hz / _RESOLUTION_SAFETY_FACTOR
    detected = bool(np.isfinite(headline_db)) and (
        headline_db > _DETECTION_NOISE_FLOOR_DB
    )

    if detected:
        reason = "detected"
    else:
        # Spectrum physical range. compute_fft_spectrum is one-sided by
        # default → freqs ∈ [0, Nyquist]; freqs[-1] gives the upper
        # bound. The lower bound is freqs[0] (≥ 0 in one-sided mode).
        f_min = float(freqs[0]) if len(freqs) > 0 else 0.0
        f_max = float(freqs[-1]) if len(freqs) > 0 else 0.0
        all_out_of_range = all(
            f < f_min or f > f_max for f in expected_sideband_freqs_hz
        )
        if all_out_of_range:
            reason = "frequency_out_of_range"
        elif brb_sideband_distance_hz is not None:
            # Hann main-lobe half-width estimate at the native bin width
            # (`T_signal ≈ 1 / bin_width` → first null at `2/T = 2·bw`).
            # Conservative: assumes no zero-padding. With heavy padding
            # the estimate over-reports the lobe width, but in that
            # regime the spectrum has enough resolution that the lobe
            # has already shrunk in absolute terms, so the case
            # typically resolves to ``detected`` and never reaches this
            # branch.
            main_lobe_half_hz = 2.0 * fft_bin_width_hz
            if brb_sideband_distance_hz < main_lobe_half_hz:
                reason = "sideband_inside_supply_main_lobe"
            elif fft_bin_width_hz > min_bin_width:
                reason = "frequency_resolution_insufficient"
            else:
                reason = "no_sideband_present"
        elif fft_bin_width_hz > min_bin_width:
            reason = "frequency_resolution_insufficient"
        else:
            reason = "no_sideband_present"

    return {
        "detected": detected,
        "reason": reason,
        "fft_bin_width_hz": round(fft_bin_width_hz, 6),
        "tolerance_hz": float(tolerance_hz),
        "min_bin_width_for_tolerance_hz": round(min_bin_width, 6),
    }


def _classify_severity(db_value: float, thresholds: dict[str, float]) -> str:
    """Classify severity from dB value and ordered thresholds."""
    if db_value <= thresholds["healthy"]:
        return "healthy"
    elif db_value <= thresholds["incipient"]:
        return "incipient"
    elif db_value <= thresholds["moderate"]:
        return "moderate"
    else:
        return "severe"


# ---------------------------------------------------------------------------
# Broken Rotor Bars
# ---------------------------------------------------------------------------

def brb_fault_index(
    freqs: NDArray[np.floating],
    amps: NDArray[np.floating],
    params: MotorParameters,
    tolerance_hz: float = 0.5,
) -> dict:
    """Compute the Broken Rotor Bar (BRB) fault index.

    The index is the ratio of the lower and upper sideband amplitudes
    at (1 ± 2s)·f_s to the fundamental amplitude, expressed in dB.

    Args:
        freqs: Frequency axis of the spectrum.
        amps: Amplitude values of the spectrum.
        params: Motor parameters (for slip and supply frequency).
        tolerance_hz: Frequency search tolerance.

    Returns:
        Dictionary with frequencies found, amplitudes, dB indices,
        and severity classification.
    """
    fs = params.supply_freq_hz
    s = params.slip

    f_lower = (1 - 2 * s) * fs
    f_upper = (1 + 2 * s) * fs

    fundamental = amplitude_at_frequency(freqs, amps, fs, tolerance_hz)
    lower_sb = amplitude_at_frequency(freqs, amps, f_lower, tolerance_hz)
    upper_sb = amplitude_at_frequency(freqs, amps, f_upper, tolerance_hz)

    a_fund = fundamental["amplitude"]
    a_lower = lower_sb["amplitude"]
    a_upper = upper_sb["amplitude"]

    db_lower = _db_ratio(a_lower, a_fund)
    db_upper = _db_ratio(a_upper, a_fund)
    db_combined = _db_ratio((a_lower + a_upper) / 2.0, a_fund) if a_fund > 0 else -np.inf

    severity = _classify_severity(max(db_lower, db_upper), BRB_THRESHOLDS)

    return {
        "fault_type": "broken_rotor_bars",
        "fundamental": {
            "expected_hz": fs,
            **fundamental,
        },
        "lower_sideband": {
            "expected_hz": round(f_lower, 4),
            **lower_sb,
            "db_relative": round(float(db_lower), 2),
        },
        "upper_sideband": {
            "expected_hz": round(f_upper, 4),
            **upper_sb,
            "db_relative": round(float(db_upper), 2),
        },
        "combined_index_db": round(float(db_combined), 2),
        "severity": severity,
        "thresholds_db": BRB_THRESHOLDS,
        "detection_status": _build_detection_status(
            freqs=freqs,
            tolerance_hz=tolerance_hz,
            headline_db=float(db_combined),
            expected_sideband_freqs_hz=[f_lower, f_upper],
            brb_sideband_distance_hz=2.0 * s * fs,
        ),
    }


# ---------------------------------------------------------------------------
# Eccentricity
# ---------------------------------------------------------------------------

def eccentricity_fault_index(
    freqs: NDArray[np.floating],
    amps: NDArray[np.floating],
    params: MotorParameters,
    harmonics: int = 3,
    tolerance_hz: float = 0.5,
) -> dict:
    """Compute eccentricity fault indices.

    Searches for sidebands at f_s ± k·f_r (k = 1 … harmonics).

    Args:
        freqs: Frequency axis.
        amps: Amplitude values.
        params: Motor parameters.
        harmonics: Number of harmonic orders.
        tolerance_hz: Frequency tolerance.

    Returns:
        Dictionary with sideband amplitudes, dB indices, severity.
    """
    fs = params.supply_freq_hz
    fr = params.rotor_freq_hz
    fund = amplitude_at_frequency(freqs, amps, fs, tolerance_hz)
    a_fund = fund["amplitude"]

    sidebands = []
    worst_db = -np.inf

    for k in range(1, harmonics + 1):
        f_lo = fs - k * fr
        f_hi = fs + k * fr
        sb_lo = amplitude_at_frequency(freqs, amps, f_lo, tolerance_hz)
        sb_hi = amplitude_at_frequency(freqs, amps, f_hi, tolerance_hz)

        db_lo = _db_ratio(sb_lo["amplitude"], a_fund)
        db_hi = _db_ratio(sb_hi["amplitude"], a_fund)

        worst_db = max(worst_db, db_lo, db_hi)

        sidebands.append({
            "harmonic_order": k,
            "lower": {
                "expected_hz": round(f_lo, 4),
                **sb_lo,
                "db_relative": round(float(db_lo), 2),
            },
            "upper": {
                "expected_hz": round(f_hi, 4),
                **sb_hi,
                "db_relative": round(float(db_hi), 2),
            },
        })

    severity = _classify_severity(float(worst_db), ECCENTRICITY_THRESHOLDS)

    expected_sidebands_for_status: list[float] = []
    for k in range(1, harmonics + 1):
        expected_sidebands_for_status.extend([fs - k * fr, fs + k * fr])

    return {
        "fault_type": "eccentricity",
        "fundamental": {
            "expected_hz": fs,
            **fund,
        },
        "sidebands": sidebands,
        "worst_sideband_db": round(float(worst_db), 2),
        "severity": severity,
        "thresholds_db": ECCENTRICITY_THRESHOLDS,
        "detection_status": _build_detection_status(
            freqs=freqs,
            tolerance_hz=tolerance_hz,
            headline_db=float(worst_db),
            expected_sideband_freqs_hz=expected_sidebands_for_status,
        ),
    }


# ---------------------------------------------------------------------------
# Stator inter-turn short circuit
# ---------------------------------------------------------------------------

def stator_fault_index(
    freqs: NDArray[np.floating],
    amps: NDArray[np.floating],
    params: MotorParameters,
    harmonics: int = 3,
    tolerance_hz: float = 0.5,
) -> dict:
    """Compute stator inter‑turn fault indices.

    Looks for sidebands at f_s ± 2k·f_r.

    Args:
        freqs: Frequency axis.
        amps: Amplitude values.
        params: Motor parameters.
        harmonics: Number of harmonic orders.
        tolerance_hz: Frequency tolerance.

    Returns:
        Dictionary with sideband analysis and severity.
    """
    fs = params.supply_freq_hz
    fr = params.rotor_freq_hz
    fund = amplitude_at_frequency(freqs, amps, fs, tolerance_hz)
    a_fund = fund["amplitude"]

    sidebands = []
    worst_db = -np.inf

    for k in range(1, harmonics + 1):
        f_lo = fs - 2 * k * fr
        f_hi = fs + 2 * k * fr
        sb_lo = amplitude_at_frequency(freqs, amps, f_lo, tolerance_hz)
        sb_hi = amplitude_at_frequency(freqs, amps, f_hi, tolerance_hz)

        db_lo = _db_ratio(sb_lo["amplitude"], a_fund)
        db_hi = _db_ratio(sb_hi["amplitude"], a_fund)
        worst_db = max(worst_db, db_lo, db_hi)

        sidebands.append({
            "harmonic_order": k,
            "lower": {
                "expected_hz": round(f_lo, 4),
                **sb_lo,
                "db_relative": round(float(db_lo), 2),
            },
            "upper": {
                "expected_hz": round(f_hi, 4),
                **sb_hi,
                "db_relative": round(float(db_hi), 2),
            },
        })

    severity = _classify_severity(float(worst_db), ECCENTRICITY_THRESHOLDS)

    return {
        "fault_type": "stator_inter_turn",
        "fundamental": {
            "expected_hz": fs,
            **fund,
        },
        "sidebands": sidebands,
        "worst_sideband_db": round(float(worst_db), 2),
        "severity": severity,
        "thresholds_db": ECCENTRICITY_THRESHOLDS,
    }


# ---------------------------------------------------------------------------
# Bearing faults (via stator current)
# ---------------------------------------------------------------------------

def bearing_fault_index(
    freqs: NDArray[np.floating],
    amps: NDArray[np.floating],
    supply_freq_hz: float,
    bearing_defect_freq_hz: float,
    defect_type: str = "bpfo",
    harmonics: int = 2,
    tolerance_hz: float = 0.5,
) -> dict:
    """Compute bearing fault indices from stator‑current spectrum.

    Bearing defects produce torque oscillations that modulate the current,
    creating sidebands at f_s ± k · f_defect.

    Args:
        freqs: Frequency axis.
        amps: Amplitude values.
        supply_freq_hz: Supply frequency in Hz.
        bearing_defect_freq_hz: Characteristic defect frequency in Hz
            (BPFO, BPFI, BSF, or FTF).
        defect_type: Label for the defect type.
        harmonics: Number of sideband orders.
        tolerance_hz: Frequency tolerance.

    Returns:
        Dictionary with sideband analysis.
    """
    fs = supply_freq_hz
    fd = bearing_defect_freq_hz
    fund = amplitude_at_frequency(freqs, amps, fs, tolerance_hz)
    a_fund = fund["amplitude"]

    sidebands = []
    worst_db = -np.inf

    for k in range(1, harmonics + 1):
        f_lo = fs - k * fd
        f_hi = fs + k * fd
        sb_lo = amplitude_at_frequency(freqs, amps, f_lo, tolerance_hz)
        sb_hi = amplitude_at_frequency(freqs, amps, f_hi, tolerance_hz)

        db_lo = _db_ratio(sb_lo["amplitude"], a_fund)
        db_hi = _db_ratio(sb_hi["amplitude"], a_fund)
        worst_db = max(worst_db, db_lo, db_hi)

        sidebands.append({
            "order": k,
            "lower": {
                "expected_hz": round(f_lo, 4),
                **sb_lo,
                "db_relative": round(float(db_lo), 2),
            },
            "upper": {
                "expected_hz": round(f_hi, 4),
                **sb_hi,
                "db_relative": round(float(db_hi), 2),
            },
        })

    expected_sidebands_for_status: list[float] = []
    for k in range(1, harmonics + 1):
        expected_sidebands_for_status.extend([fs - k * fd, fs + k * fd])

    return {
        "fault_type": f"bearing_{defect_type}",
        "defect_frequency_hz": round(fd, 4),
        "fundamental": {
            "expected_hz": fs,
            **fund,
        },
        "sidebands": sidebands,
        "worst_sideband_db": round(float(worst_db), 2),
        "note": (
            "Bearing signatures in stator current are typically weak. "
            "Confirm with envelope analysis or vibration measurements."
        ),
        "detection_status": _build_detection_status(
            freqs=freqs,
            tolerance_hz=tolerance_hz,
            headline_db=float(worst_db),
            expected_sideband_freqs_hz=expected_sidebands_for_status,
        ),
    }


# ---------------------------------------------------------------------------
# Band energy index (cavitation, load faults)
# ---------------------------------------------------------------------------

def band_energy_index(
    freqs: NDArray[np.floating],
    psd: NDArray[np.floating],
    centre_freq_hz: float,
    bandwidth_hz: float = 5.0,
) -> dict:
    """Compute the integrated spectral energy in a frequency band.

    Useful as generic fault/cavitation indicator: the energy in a band
    around the supply frequency or other characteristic region.

    Args:
        freqs: Frequency axis (from PSD).
        psd: PSD values.
        centre_freq_hz: Centre of the integration band.
        bandwidth_hz: Total bandwidth for integration.

    Returns:
        Dictionary with band energy, limits used.
    """
    low = centre_freq_hz - bandwidth_hz / 2.0
    high = centre_freq_hz + bandwidth_hz / 2.0

    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return {
            "centre_freq_hz": centre_freq_hz,
            "bandwidth_hz": bandwidth_hz,
            "band_energy": 0.0,
            "found": False,
        }

    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    energy = float(np.sum(psd[mask]) * df)

    return {
        "centre_freq_hz": centre_freq_hz,
        "bandwidth_hz": bandwidth_hz,
        "band_low_hz": round(low, 4),
        "band_high_hz": round(high, 4),
        "band_energy": energy,
        "found": True,
    }


# ---------------------------------------------------------------------------
# Statistical indices on envelope
# ---------------------------------------------------------------------------

def envelope_statistical_indices(
    envelope: NDArray[np.floating],
) -> dict:
    """Compute statistical indices of the envelope signal.

    Kurtosis, skewness, crest factor, and RMS — indicators of impulsive
    content from bearing or gear faults.

    Args:
        envelope: Amplitude envelope of the current signal.

    Returns:
        Dictionary of statistical indices.
    """
    from scipy.stats import kurtosis, skew

    rms = float(np.sqrt(np.mean(envelope ** 2)))
    peak = float(np.max(np.abs(envelope)))
    crest = peak / rms if rms > 0 else 0.0

    return {
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "crest_factor": round(crest, 4),
        "kurtosis": round(float(kurtosis(envelope, fisher=True)), 4),
        "skewness": round(float(skew(envelope)), 4),
    }
