"""Motor parameter calculations for MCSA.

Computes synchronous speed, slip, rotor frequency, and expected fault
frequencies from motor nameplate data and operating conditions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MotorParameters:
    """Computed operating parameters for an induction motor.

    Attributes:
        supply_freq_hz: Supply (line) frequency in Hz.
        poles: Number of magnetic poles.
        sync_speed_rpm: Synchronous speed in RPM.
        rotor_speed_rpm: Measured rotor speed in RPM.
        slip: Per-unit slip (0–1).
        rotor_freq_hz: Mechanical rotational frequency in Hz.
        slip_freq_hz: Slip frequency in Hz (s × f_supply).
    """

    supply_freq_hz: float
    poles: int
    sync_speed_rpm: float
    rotor_speed_rpm: float
    slip: float
    rotor_freq_hz: float
    slip_freq_hz: float

    def to_dict(self) -> dict:
        return asdict(self)


def calculate_motor_parameters(
    supply_freq_hz: float,
    poles: int,
    rotor_speed_rpm: float,
) -> MotorParameters:
    """Calculate motor operating parameters from nameplate / measured data.

    Args:
        supply_freq_hz: Supply frequency in Hz (e.g. 50 or 60).
        poles: Number of poles (must be even, ≥ 2).
        rotor_speed_rpm: Measured rotor speed in RPM.

    Returns:
        MotorParameters with all derived quantities.

    Raises:
        ValueError: If inputs are physically invalid.
    """
    if supply_freq_hz <= 0:
        raise ValueError(f"Supply frequency must be > 0, got {supply_freq_hz}")
    if poles < 2 or poles % 2 != 0:
        raise ValueError(f"Poles must be even and ≥ 2, got {poles}")
    if rotor_speed_rpm < 0:
        raise ValueError(f"Rotor speed must be ≥ 0, got {rotor_speed_rpm}")

    sync_speed_rpm = 120.0 * supply_freq_hz / poles

    if rotor_speed_rpm > sync_speed_rpm:
        raise ValueError(
            f"Rotor speed ({rotor_speed_rpm} RPM) exceeds synchronous speed "
            f"({sync_speed_rpm} RPM) — not valid for a motor (generator mode)."
        )

    slip = (sync_speed_rpm - rotor_speed_rpm) / sync_speed_rpm
    rotor_freq_hz = rotor_speed_rpm / 60.0
    slip_freq_hz = slip * supply_freq_hz

    return MotorParameters(
        supply_freq_hz=supply_freq_hz,
        poles=poles,
        sync_speed_rpm=sync_speed_rpm,
        rotor_speed_rpm=rotor_speed_rpm,
        slip=slip,
        rotor_freq_hz=rotor_freq_hz,
        slip_freq_hz=slip_freq_hz,
    )


def calculate_fault_frequencies(
    params: MotorParameters,
    harmonics: int = 3,
) -> dict:
    """Calculate expected fault frequencies for common induction‑motor faults.

    Computes characteristic harmonic frequencies for:
      - Broken Rotor Bars (BRB): sidebands at (1 ± 2ks)·f_s
      - Eccentricity (static/dynamic): f_s ± k·f_r
      - Stator inter‑turn faults: f_s ± 2k·f_r
      - Mixed eccentricity: n·f_r (for n = 1 … harmonics)

    Args:
        params: Motor operating parameters.
        harmonics: Number of harmonic orders to compute (default 3).

    Returns:
        Dictionary with fault type keys → lists of expected frequencies [Hz].
    """
    fs = params.supply_freq_hz
    s = params.slip
    fr = params.rotor_freq_hz

    # Broken Rotor Bars — sidebands (1 ± 2ks)·fs
    brb_lower = [(1 - 2 * k * s) * fs for k in range(1, harmonics + 1)]
    brb_upper = [(1 + 2 * k * s) * fs for k in range(1, harmonics + 1)]

    # Eccentricity — fs ± k·fr
    ecc_freqs = []
    for k in range(1, harmonics + 1):
        ecc_freqs.append({"harmonic_order": k, "lower": fs - k * fr, "upper": fs + k * fr})

    # Stator inter‑turn — fs ± 2k·fr
    stator_freqs = []
    for k in range(1, harmonics + 1):
        stator_freqs.append({"harmonic_order": k, "lower": fs - 2 * k * fr, "upper": fs + 2 * k * fr})

    # Mixed eccentricity — n·fr
    mixed_ecc = [k * fr for k in range(1, harmonics + 1)]

    return {
        "motor_parameters": params.to_dict(),
        "broken_rotor_bars": {
            "description": "Sidebands at (1 ± 2k·s)·f_supply due to rotor asymmetry",
            "lower_sidebands_hz": brb_lower,
            "upper_sidebands_hz": brb_upper,
        },
        "eccentricity": {
            "description": "Sidebands at f_supply ± k·f_rotor due to air‑gap non‑uniformity",
            "sidebands": ecc_freqs,
        },
        "stator_faults": {
            "description": "Sidebands at f_supply ± 2k·f_rotor due to stator winding asymmetry",
            "sidebands": stator_freqs,
        },
        "mixed_eccentricity": {
            "description": "Components at n·f_rotor (pure rotational harmonics)",
            "frequencies_hz": mixed_ecc,
        },
    }
