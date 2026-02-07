# MCSA Server — Usage Guide

> **Motor Current Signature Analysis via the Model Context Protocol**
>
> A comprehensive signal-processing toolkit that turns any MCP-compatible
> AI assistant into an electric-motor condition-monitoring expert.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Data Store](#data-store)
4. [Server Capabilities](#server-capabilities)
5. [Tool Reference](#tool-reference)
   - [Signal Acquisition](#signal-acquisition)
   - [Motor Parameters](#motor-parameters)
   - [Signal Preprocessing](#signal-preprocessing)
   - [Spectral Analysis](#spectral-analysis)
   - [Fault Detection](#fault-detection)
   - [Envelope & Time-Frequency Analysis](#envelope--time-frequency-analysis)
   - [One-Shot Diagnostic Pipelines](#one-shot-diagnostic-pipelines)
   - [Data Store Management](#data-store-management)
6. [Resources](#resources)
7. [Prompts](#prompts)
8. [Diagnostic Workflows](#diagnostic-workflows)
9. [Working with Real Signals](#working-with-real-signals)
10. [Integration Patterns](#integration-patterns)
11. [Severity Classification](#severity-classification)
12. [FAQ](#faq)

---

## Overview

`mcp-server-mcsa` exposes **21 tools**, **1 resource**, and
**1 prompt** over the
[Model Context Protocol](https://modelcontextprotocol.io) (MCP).
An LLM-based host (Claude Desktop, VS Code Copilot, or any MCP client)
can call these tools to:

| Capability | Description |
|---|---|
| Load measured signals | CSV, WAV, NumPy `.npy` — directly from field acquired data |
| Compute motor parameters | Slip, synchronous speed, rotor frequency |
| Predict fault frequencies | BRB, eccentricity, stator, bearing defect markers |
| Preprocess | DC removal, windowing, bandpass / notch / lowpass filtering |
| Analyse spectra | FFT, PSD, peak detection |
| Detect faults | Severity-classified indices for BRB, eccentricity, stator, bearing |
| Envelope analysis | Hilbert envelope, instantaneous frequency, envelope spectrum |
| Time-frequency | STFT, spectrogram, frequency tracking |
| Run complete diagnostics | One-shot pipeline from signal array or from file |
| Manage stored data | List, inspect, and clear persisted signals and spectra |

Signals and spectra are **persisted to disk** (`~/.mcsa_data/`) and
referenced by short IDs (`sig_xxxx`, `spec_xxxx`). Raw arrays never
enter the chat context.

---

## Architecture

```
┌─────────────────────────────────────┐
│          MCP Host / LLM             │
│  (Claude Desktop, VS Code, etc.)    │
└──────────────┬──────────────────────┘
               │  MCP (stdio / SSE)
┌──────────────▼──────────────────────┐
│       mcp-server-mcsa               │
│  ┌──────────┐  ┌──────────────────┐ │
│  │ FastMCP  │  │  19 Tools        │ │
│  │ Server   │──│  1 Resource      │ │
│  │          │  │  1 Prompt        │ │
│  └──────────┘  └──────────────────┘ │
│        │                            │
│  ┌─────▼──────────────────────────┐ │
│  │  analysis/                     │ │
│  │  motor · bearing · spectral    │ │
│  │  preprocessing · envelope      │ │
│  │  timefreq · fault_detection    │ │
│  │  test_signal · file_io         │ │
│  └────────────────────────────────┘ │
└─────────────────────────────────────┘
```

The **analysis** layer is a pure-Python library (NumPy + SciPy) with no
dependency on the MCP SDK — it can be imported directly in your own
scripts if needed.

---

## Server Capabilities

| Feature | Details |
|---|---|
| Transport | `stdio` (default), extensible to SSE/WebSocket |
| Protocol | MCP 1.0+ |
| Runtime | Python ≥ 3.10 |
| Dependencies | `numpy`, `scipy`, `pydantic`, `mcp` |
| Deployment | `pip install`, Docker, or direct `uv run` |

---

## Tool Reference

### Signal Acquisition

#### `inspect_signal_file`

Inspect a signal file **without loading** it.  Returns format detection,
file size, column names / WAV metadata / NPY shape.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `file_path` | `str` | ✔ | Absolute path to the file |

**Returns** — JSON with `exists`, `detected_format`, `file_size_bytes`,
and format-specific details (`csv_details`, `wav_details`, or
`npy_details`).

---

#### `load_signal_from_file`

Load a current signal from a **CSV / TSV / WAV / NPY** file.
The signal is **persisted to disk** and a `signal_id` is returned.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `file_path` | `str` | ✔ | | Absolute path |
| `sampling_freq_hz` | `float` | for NPY | `null` | Sampling rate (auto-detected for WAV; inferred from time column for CSV) |
| `signal_column` | `int \| str` | | `1` | CSV: column index or header name |
| `time_column` | `int \| str \| null` | | `0` | CSV: time column index/name (`null` if absent) |
| `channel` | `int` | | `0` | WAV: channel index |

**Returns** — JSON with `signal_id`, `signal_summary` (n_samples,
duration_s, rms, peak_amplitude), `sampling_freq_hz`, `format`,
and `file_path`.

---

#### `generate_test_current_signal`

Generate a synthetic motor-current signal with optional injected faults.
Ideal for testing, demonstrations, and benchmarking.
The signal is **persisted to disk** and a `signal_id` is returned.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `supply_freq_hz` | `float` | — | Supply frequency |
| `poles` | `int` | — | Number of poles |
| `rotor_speed_rpm` | `float` | — | Rotor speed |
| `sampling_freq_hz` | `float` | `10000` | Sampling rate |
| `duration_s` | `float` | `2.0` | Duration |
| `snr_db` | `float` | `40` | Signal-to-noise ratio |
| `fault_type` | `str \| null` | `null` | `brb`, `eccentricity`, `bearing`, or `null` (healthy) |
| `fault_severity` | `float` | `0.05` | 0–1 scale |
| `bearing_defect_freq_hz` | `float` | `120.0` | Bearing defect frequency |

---

### Motor Parameters

#### `calculate_motor_params`

Compute synchronous speed, slip, and rotor frequency from nameplate data.

| Parameter | Type | Description |
|---|---|---|
| `supply_freq_hz` | `float` | Grid / inverter supply frequency |
| `poles` | `int` | Number of poles |
| `rotor_speed_rpm` | `float` | Measured rotor speed |

**Returns** — `synchronous_speed_rpm`, `slip`, `rotor_frequency_hz`,
`slip_frequency_hz`, plus all inputs echoed.

---

#### `compute_fault_frequencies`

Predict the characteristic spectral lines of common faults.

| Parameter | Type | Description |
|---|---|---|
| `supply_freq_hz` | `float` | Supply frequency |
| `poles` | `int` | Number of poles |
| `rotor_speed_rpm` | `float` | Rotor speed |

**Returns** — Dictionary with `brb_sidebands`, `eccentricity_sidebands`,
`stator_harmonics`, and motor parameters.

---

#### `compute_bearing_frequencies`

Compute BPFO, BPFI, BSF, FTF, and their current sidebands from bearing
geometry.

| Parameter | Type | Description |
|---|---|---|
| `n_balls` | `int` | Number of rolling elements |
| `ball_diameter_mm` | `float` | Ball/roller diameter |
| `pitch_diameter_mm` | `float` | Pitch circle diameter |
| `contact_angle_deg` | `float` | Contact angle in degrees |
| `rotor_speed_rpm` | `float` | Shaft speed |
| `supply_freq_hz` | `float` | Supply frequency |

---

### Signal Preprocessing

#### `preprocess_signal`

Multi-step signal conditioning pipeline.
Accepts a `signal_id` or raw array. Returns a **new `signal_id`** for
the preprocessed signal.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `signal_id` | `str \| null` | `null` | ID of stored signal (preferred) |
| `signal` | `list[float]` | — | Raw signal (fallback) |
| `sampling_freq_hz` | `float` | — | Sampling rate (auto from signal_id) |
| `remove_dc` | `bool` | `true` | Remove DC offset |
| `window` | `str` | `hann` | Window function |
| `filter_type` | `str \| null` | `null` | `bandpass`, `lowpass`, `notch` |
| `low_cut_hz` | `float` | `1.0` | Bandpass lower bound |
| `high_cut_hz` | `float` | `2000.0` | Bandpass upper bound |
| `notch_freq_hz` | `float` | `50.0` | Notch centre frequency |
| `notch_q` | `float` | `30.0` | Notch quality factor |

---

### Spectral Analysis

#### `compute_spectrum`

Compute the one- or two-sided FFT amplitude spectrum.
Returns a `spectrum_id` and compact summary (top peaks).

| Parameter | Type | Default |
|---|---|---|
| `signal_id` | `str \| null` | `null` |
| `signal` | `list[float]` | — |
| `sampling_freq_hz` | `float` | — |
| `sided` | `str` | `one` |
| `normalize` | `bool` | `true` |

---

#### `compute_power_spectral_density`

Welch-method power spectral density (averaged, reduced variance).
Returns a `spectrum_id` and compact summary.

| Parameter | Type | Default |
|---|---|---|
| `signal_id` | `str \| null` | `null` |
| `signal` | `list[float]` | — |
| `sampling_freq_hz` | `float` | — |
| `nperseg` | `int \| null` | `null` |
| `noverlap` | `int \| null` | `null` |

---

#### `find_spectrum_peaks`

Peak detection with amplitude and prominence thresholds.
Accepts a `spectrum_id` or raw arrays.

| Parameter | Type | Default |
|---|---|---|
| `spectrum_id` | `str \| null` | `null` |
| `frequencies` | `list[float]` | — |
| `amplitudes` | `list[float]` | — |
| `prominence` | `float` | `0.001` |
| `max_peaks` | `int` | `20` |
| `min_freq_hz` | `float` | `0.0` |
| `max_freq_hz` | `float \| null` | `null` |

---

### Fault Detection

All fault-detection tools return a JSON dictionary with:

- **`severity`** — `healthy`, `incipient`, `moderate`, or `severe`
- **`index_db`** — principal fault index (dB relative to fundamental)
- Per-sideband amplitudes, frequencies, and descriptions

#### `detect_broken_rotor_bars`

Checks the lower/upper sidebands at $(1 \pm 2ks) \cdot f_s$.

| Parameter | Type |
|---|---|
| `spectrum_id` | stored spectrum (preferred) |
| `frequencies`, `amplitudes` | spectrum arrays (fallback) |
| `supply_freq_hz`, `poles`, `rotor_speed_rpm` | motor nameplate |
| `tolerance_hz` | search window (default `0.5`) |

---

#### `detect_eccentricity`

Sidebands at $f_s \pm k \cdot f_r$ (static/dynamic air-gap eccentricity).

---

#### `detect_stator_faults`

Sidebands at $f_s \pm 2k \cdot f_r$ (inter-turn short circuits).

---

#### `detect_bearing_faults`

Requires the bearing defect frequency; analyses
$f_s \pm k \cdot f_{\text{defect}}$.

| Extra parameter | Type |
|---|---|
| `bearing_defect_freq_hz` | `float` |
| `bearing_type` | `str` — label for the report |

---

#### `compute_band_energy`

Energy ratio in a band around the fundamental vs. total spectral energy.

---

### Envelope & Time-Frequency Analysis

#### `compute_envelope_spectrum`

Hilbert-transform envelope → FFT to reveal low-frequency modulations
caused by bearing defects and mechanical looseness.

---

#### `compute_time_frequency`

Short-Time Fourier Transform (STFT) spectrogram with optional frequency
tracking — useful for ramp-up/ramp-down transient analysis.

| Extra parameter | Type | Description |
|---|---|---|
| `track_freq_hz` | `float \| null` | Frequency to track over time |
| `track_tolerance_hz` | `float` | Search band half-width |

---

### One-Shot Diagnostic Pipelines

#### `run_full_diagnosis`

Pass a `signal_id` (or raw signal array) + motor nameplate → get a complete diagnostic report
(spectrum, all fault indices, envelope statistics, severity summary).

---

#### `diagnose_from_file`

Same as above, but reads the signal directly from a file path.
The loaded signal is also stored and a `signal_id` is included in the
report for follow-up analysis.

| Unique to this tool | Type | Description |
|---|---|---|
| `file_path` | `str` | Path to CSV / WAV / NPY |
| All motor params | — | As in `run_full_diagnosis` |
| All file params | — | As in `load_signal_from_file` |

**Returns** — Same comprehensive JSON as `run_full_diagnosis`, plus
`signal_id`, `source_file` and `file_format` fields.

---

### Data Store Management

#### `list_stored_data`

List all signals and spectra currently persisted on disk.
Returns a compact summary per item (ID, type, size, duration) —
never the raw arrays.

---

#### `clear_stored_data`

Delete stored items from disk and memory.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data_id` | `str \| null` | `null` | Specific ID to remove, or omit to clear all |

---

## Resources

### `mcsa://fault-signatures`

A read-only knowledge base listing the characteristic spectral signatures
of the four major fault families:

| Fault | Signature Formula |
|---|---|
| Broken Rotor Bars | $(1 \pm 2ks) \cdot f_s$ |
| Eccentricity | $f_s \pm k \cdot f_r$ |
| Stator Inter-Turn | $f_s \pm 2k \cdot f_r$ |
| Bearing Defects | $f_s \pm k \cdot f_{\text{defect}}$ |

Access it via `mcp.read_resource("mcsa://fault-signatures")` from your
MCP client.

---

## Prompts

### `analyze_motor_current`

A guided, step-by-step prompt that walks the LLM through a complete
diagnostic session.  Parameters:

| Parameter | Default |
|---|---|
| `motor_type` | `induction` |
| `supply_freq_hz` | `50` |
| `poles` | `4` |
| `rotor_speed_rpm` | `1470` |

The prompt instructs the assistant to:

1. Load a measured signal (or generate a synthetic one)
2. Calculate motor parameters and fault frequencies
3. Preprocess the signal
4. Compute the spectrum
5. Run all fault detectors
6. Perform envelope analysis
7. Compile a severity-classified report with recommendations

---

## Diagnostic Workflows

### Workflow A — Synthetic Signal (Demo / Benchmarking)

```
generate_test_current_signal       → sig_0001
  → preprocess_signal(sig_0001)    → sig_0002
    → compute_spectrum(sig_0002)   → spec_0001
      → detect_broken_rotor_bars(spec_0001)
      → detect_eccentricity(spec_0001)
      → ...
```

### Workflow B — Real Signal from File

```
inspect_signal_file                    (check format, columns, metadata)
  → load_signal_from_file              → sig_0001
    → preprocess_signal(sig_0001)      → sig_0002
      → compute_spectrum(sig_0002)     → spec_0001
        → detect_broken_rotor_bars(spec_0001)
        → detect_eccentricity(spec_0001)
        → detect_stator_faults(spec_0001)
        → detect_bearing_faults(spec_0001)
          → compute_envelope_spectrum(sig_0002)
            → compute_time_frequency(sig_0002)   (optional)
```

### Workflow C — One-Shot from File

```
diagnose_from_file           (single call: load → preprocess → analyse → report)
```

### Workflow D — Guided Interactive Session

```
Start the `analyze_motor_current` prompt
  → LLM walks through the complete pipeline step-by-step
```

---

## Working with Real Signals

### Supported File Formats

| Format | Extensions | Sampling Rate | Notes |
|---|---|---|---|
| CSV / TSV | `.csv`, `.tsv`, `.txt` | Inferred from time column or provided explicitly | Header row auto-detected; configurable column indices/names |
| WAV | `.wav` | Embedded in header | 8 / 16 / 24 / 32-bit integer; mono or multi-channel |
| NumPy | `.npy` | Must be provided | 1-D or 2-D (first column used) |

### Data Acquisition Tips

- **Sampling rate**: Nyquist requires $f_s \geq 2 f_{\max}$. For typical
  MCSA up to the 5th supply harmonic at 50 Hz, $f_s \geq 4{,}000$ Hz is
  sufficient, but 8–10 kHz is recommended.
- **Duration**: At least 5–10 seconds of steady-state operation to achieve
  adequate spectral resolution ($\Delta f = 1/T$).
- **Sensor**: Hall-effect current transducer or Rogowski coil on one
  stator phase. Ensure the sensor bandwidth covers the frequency range
  of interest.
- **Anti-aliasing**: Use an analogue low-pass filter with a cut-off below
  $f_s / 2$ before digitisation.

### Example: Diagnosing from a CSV File

Suppose you measured the phase-A current of a 4-pole, 50 Hz induction
motor at 1470 RPM and saved it to `motor_phase_a.csv` with columns
`time,current`:

```
User:  Diagnose the motor from this file:
       C:\measurements\motor_phase_a.csv
       50 Hz supply, 4 poles, 1470 RPM

LLM → calls diagnose_from_file(
         file_path="C:\\measurements\\motor_phase_a.csv",
         supply_freq_hz=50, poles=4, rotor_speed_rpm=1470)

LLM ← receives complete diagnostic JSON:
       BRB severity:         healthy
       Eccentricity severity: incipient
       Stator severity:      healthy
       → Recommendation: schedule vibration check for air-gap alignment
```

---

## Integration Patterns

### Claude Desktop

In `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "mcsa": {
      "command": "uvx",
      "args": ["mcp-server-mcsa"]
    }
  }
}
```

### VS Code (GitHub Copilot)

In `.vscode/mcp.json`:

```json
{
  "servers": {
    "mcsa": {
      "command": "uvx",
      "args": ["mcp-server-mcsa"]
    }
  }
}
```

### Python Script (Direct Library Use)

The analysis modules can be imported directly without the MCP layer:

```python
from mcp_server_mcsa.analysis.file_io import load_signal
from mcp_server_mcsa.analysis.preprocessing import preprocess_pipeline
from mcp_server_mcsa.analysis.spectral import compute_fft_spectrum
from mcp_server_mcsa.analysis.fault_detection import brb_fault_index
from mcp_server_mcsa.analysis.motor import calculate_motor_parameters

# Load measured signal
data = load_signal("measurement.csv")
signal = data["signal"]
fs = data["sampling_freq_hz"]

# Motor parameters
params = calculate_motor_parameters(50.0, 4, 1470.0)

# Preprocess + spectrum
x = preprocess_pipeline(signal, fs, window="hann")
freqs, amps = compute_fft_spectrum(x, fs, sided="one")

# Fault detection
result = brb_fault_index(freqs, amps, params)
print(result["severity"])  # 'healthy' | 'incipient' | 'moderate' | 'severe'
```

### Docker

```bash
docker build -t mcp-server-mcsa .
docker run -i --rm mcp-server-mcsa
```

In the host's MCP configuration, use `"command": "docker"` with
`"args": ["run", "-i", "--rm", "mcp-server-mcsa"]`.

---

## Severity Classification

All fault-detection tools classify results into four levels based on the
sideband-to-fundamental amplitude ratio (in dB):

| Level | BRB Threshold | Eccentricity | Stator | Description |
|---|---|---|---|---|
| **Healthy** | > −40 dB | > −35 dB | > −40 dB | No action required |
| **Incipient** | −40 to −35 dB | −35 to −25 dB | −40 to −30 dB | Schedule inspection |
| **Moderate** | −35 to −25 dB | −25 to −15 dB | −30 to −20 dB | Plan maintenance |
| **Severe** | < −25 dB | < −15 dB | < −20 dB | Immediate intervention |

For envelope-based bearing detection, kurtosis and crest factor are the
principal indicators:

| Metric | Healthy | Incipient | Moderate | Severe |
|---|---|---|---|---|
| Kurtosis | < 3.5 | 3.5–5 | 5–8 | > 8 |
| Crest Factor | < 4 | 4–6 | 6–9 | > 9 |

---

## FAQ

**Q: Does the server store or modify any files?**
Yes — signals and spectra are persisted as compressed `.npz` files in
`~/.mcsa_data/` (configurable via `MCSA_DATA_DIR` environment variable).
This keeps large arrays out of the chat context and allows data to
survive server restarts. Use `list_stored_data` and `clear_stored_data`
to manage the stored data.

**Q: Can I use inverter-fed motors?**
Yes.  Set `supply_freq_hz` to the actual inverter output frequency.  Be
aware that variable-frequency drives introduce additional spectral
components; higher resolution (longer acquisition) may be needed.

**Q: How accurate is the fault classification?**
The thresholds are based on widely accepted values from peer-reviewed
literature (Benbouzid 2003, Nandi et al. 2005, Blodt et al. 2008) and
international standards.  Real-world accuracy depends on signal quality,
load conditions, and motor design.

**Q: What if I only have a WAV file from audio recording?**
WAV files are fully supported.  Ensure the recording captures the actual
current waveform (via a current transducer connected to an audio
interface) and note the effective sampling rate.

**Q: Can I analyse transient start-ups?**
Yes — use `compute_time_frequency` (STFT) with the `track_freq_hz`
parameter to follow fault signatures during speed ramp-up.

---

*For the full API source and contribution guidelines, see the
[README](README.md).*
