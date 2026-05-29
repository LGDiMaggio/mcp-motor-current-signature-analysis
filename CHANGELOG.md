# Changelog

All notable changes to `mcp-server-mcsa` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] — 2026-05-29

Three coordinated enhancements addressing limitations of v0.2.2 identified
empirically by the downstream [llm-mcsa-diagnosis-bench](https://github.com/LGDiMaggio/llm-mcsa-diagnosis-bench)
plan-gate spot-check on real 100%Load stator-current data. All changes are
**strictly additive** — v0.2.2 consumers see identical behaviour when they
do not pass the new keyword arguments.

### Added

- **`detect_peaks(..., interpolate: bool = False)`** (PR #4, closes #3) —
  opt-in sub-bin parabolic peak refinement in the log-magnitude domain
  (Smith, *Mathematics of the DFT*, Sec 9.3). Recovers true peak frequency
  and amplitude within a few percent of the bin width when the spectral
  peak falls between FFT bins. Default `False` preserves v0.2.2 byte-
  identical output for existing callers (incl. all three `server.py` MCP
  tools). The default is expected to flip to `True` in v0.4.0.
- **`compute_fft_spectrum(..., min_resolution_hz: float | None = None)`**
  (PR #7, closes #1) — adaptive `n_fft` zero-padding so the bin width is
  at most `min_resolution_hz / 4` (factor-4 safety margin). Lets
  downstream callers resolve sidebands within a chosen tolerance on
  short segments. Raises `ValueError` on non-positive values or requests
  that would exceed the internal safety cap (`2^22` samples ≈ 16 MB
  spectrum). Default `None` preserves v0.2.2 byte-identical output.
- **`detection_status` block in fault-index returns** (PR #6, closes #2) —
  `bearing_fault_index`, `brb_fault_index`, and `eccentricity_fault_index`
  now include a structured `detection_status` dict alongside the legacy
  return fields. Five-field schema: `detected`, `reason`,
  `fft_bin_width_hz`, `tolerance_hz`, `min_bin_width_for_tolerance_hz`.
  The `reason` field is one of `detected`, `frequency_out_of_range`,
  `sideband_inside_supply_main_lobe` (BRB only), `frequency_resolution_insufficient`,
  `no_sideband_present`. Lets consumers distinguish "no fault present"
  from "test could not be performed at this resolution" — previously
  both collapsed to `worst_sideband_db = -inf`.
- **`brb_fault_index(..., signal_duration_s: float | None = None)`** —
  optional time-domain duration so the main-lobe check can compute the
  correct Hann main-lobe half-width (`2 / T_signal`) independently of
  zero-padding. **Strongly recommended** when calling
  `compute_fft_spectrum(min_resolution_hz=...)` — without it, the
  main-lobe check falls back to a bin-width estimate that is wrong for
  zero-padded spectra and can silently classify supply-line leakage as
  a real BRB sideband on healthy signals. See migration notes below.

### Migration notes

- **Bench / downstream consumers** that use `compute_fft_spectrum(
  min_resolution_hz=X)` to recover short-segment sidebands MUST pass
  `signal_duration_s=T_segment` to `brb_fault_index(...)`. Without
  this, the BRB main-lobe check uses the post-padding bin width as a
  proxy for the time-domain window length — incorrect, and the BRB
  index can report `detection_status.detected = True` on healthy
  signals where the apparent sideband is supply-line leakage at the
  search location. The legacy fallback is preserved for v0.2.2 callers
  that don't opt into `min_resolution_hz` (where the bin-width-based
  estimate is still correct).
- **Consumers relying on bin-centred peak frequencies** (e.g.
  `peak["frequency_hz"] == 50.0` checks) should pin
  `detect_peaks(..., interpolate=False)` explicitly before v0.4.0
  flips the default to `True`. The current v0.3.0 default of `False`
  insulates existing code; the explicit pin documents intent for the
  next release.

### Backward compatibility

- All v0.2.2 legacy return-dict fields preserved bit-identically:
  `worst_sideband_db`, `combined_index_db`, `severity`, `thresholds_db`,
  `fundamental`, `sidebands`, `lower_sideband`, `upper_sideband`,
  `fault_type`, `defect_frequency_hz`, `note`. 117 tests pass on this
  release vs 93 on v0.2.2 (+24 net new tests covering the new features
  plus dedicated regression tests for the additive contract).
- All three new function-signature changes are keyword-only with
  defaults that reproduce v0.2.2 behaviour.

### Tests

- `tests/test_spectral.py` adds `TestDetectPeaksSubBinInterpolation` and
  `TestComputeFFTSpectrumAdaptive`.
- `tests/test_fault_detection.py` adds `TestBearingDetectionStatus`,
  `TestBRBDetectionStatus`, `TestEccentricityDetectionStatus`,
  `TestDetectionStatusBackwardCompat`, and
  `TestBearingFaultIndexWithAdaptiveFFT`.

### Internal

- `_RESOLUTION_SAFETY_FACTOR` shared between `spectral.py` and
  `fault_detection.py` (imported, not duplicated) — single source of
  truth prevents silent drift between the FFT engine and the
  detector's resolution check.

## [0.2.2] — 2026-02-16

- Version bump for Zenodo release citation metadata.

## [0.2.1] — 2026-02-16

- Version bump for Zenodo release.

## [0.2.0]

- Persistent disk-backed data store + 2 new tools.

## [0.1.3]

- LLM predictive-maintenance tagline added.

[0.3.0]: https://github.com/LGDiMaggio/mcp-motor-current-signature-analysis/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/LGDiMaggio/mcp-motor-current-signature-analysis/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/LGDiMaggio/mcp-motor-current-signature-analysis/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/LGDiMaggio/mcp-motor-current-signature-analysis/compare/v0.1.3...v0.2.0
