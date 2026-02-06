# mcp-server-mcsa

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-compatible-green.svg)](https://modelcontextprotocol.io)

A **Model Context Protocol (MCP) server** for **Motor Current Signature Analysis (MCSA)** — non-invasive spectral analysis and fault detection in electric motors using stator-current signals.

MCSA is an industry-standard condition-monitoring technique that analyses the harmonic content of the stator current to detect rotor, stator, bearing, and air-gap faults in electric motors — without requiring vibration sensors, downtime, or physical access to the machine.  This server brings the full MCSA diagnostic workflow to any MCP-compatible AI assistant (Claude Desktop, VS Code Copilot, and others), enabling both interactive expert analysis and automated condition-monitoring pipelines.

## Features

- **Real signal loading** — read measured data from CSV, TSV, WAV, and NumPy `.npy` files
- **Motor parameter calculation** — slip, synchronous speed, rotor frequency from nameplate data
- **Fault frequency computation** — broken rotor bars, eccentricity, stator faults, mixed eccentricity
- **Bearing defect frequencies** — BPFO, BPFI, BSF, FTF from bearing geometry
- **Signal preprocessing** — DC removal, normalisation, windowing, bandpass/notch filtering
- **Spectral analysis** — FFT spectrum, Welch PSD, spectral peak detection
- **Envelope analysis** — Hilbert-transform demodulation for mechanical/bearing faults
- **Time-frequency analysis** — STFT with frequency tracking for non-stationary conditions
- **Fault detection** — automated severity classification (healthy / incipient / moderate / severe)
- **One-shot diagnostics** — full pipeline from signal array or directly from file
- **Test signal generation** — synthetic signals with configurable fault injection for demos and benchmarking

## Tools (19)

| Tool | Description |
|------|-------------|
| `inspect_signal_file` | Inspect a signal file format and metadata without loading |
| `load_signal_from_file` | Load a current signal from CSV / WAV / NPY file |
| `calculate_motor_params` | Compute slip, sync speed, rotor frequency from motor data |
| `compute_fault_frequencies` | Calculate expected fault frequencies for all common fault types |
| `compute_bearing_frequencies` | Calculate BPFO, BPFI, BSF, FTF from bearing geometry |
| `preprocess_signal` | DC removal, filtering, normalisation, windowing pipeline |
| `compute_spectrum` | Single-sided FFT amplitude spectrum |
| `compute_power_spectral_density` | Welch PSD estimation |
| `find_spectrum_peaks` | Detect and characterise peaks in a spectrum |
| `detect_broken_rotor_bars` | BRB fault index with severity classification |
| `detect_eccentricity` | Air-gap eccentricity detection via sidebands |
| `detect_stator_faults` | Stator inter-turn short circuit detection |
| `detect_bearing_faults` | Bearing defect detection from current spectrum |
| `compute_envelope_spectrum` | Hilbert envelope spectrum for modulation analysis |
| `compute_band_energy` | Integrated spectral energy in a frequency band |
| `compute_time_frequency` | STFT analysis with optional frequency tracking |
| `generate_test_current_signal` | Synthetic motor current with optional faults |
| `run_full_diagnosis` | Complete MCSA diagnostic pipeline from signal array |
| `diagnose_from_file` | Complete MCSA diagnostic pipeline directly from file |

## Resources

| URI | Description |
|-----|-------------|
| `mcsa://fault-signatures` | Reference table of fault signatures, frequencies, and empirical thresholds |

## Prompts

| Prompt | Description |
|--------|-------------|
| `analyze_motor_current` | Step-by-step guided workflow for MCSA analysis |

## Installation

### Using uv (recommended)

```bash
uvx mcp-server-mcsa
```

### Using pip

```bash
pip install mcp-server-mcsa
```

### From source

```bash
git clone https://github.com/LGDiMaggio/mcp-motor-current-signature-analysis.git
cd mcp-motor-current-signature-analysis
pip install -e .
```

## Configuration

### Claude Desktop

Add to your Claude Desktop configuration file:

<details>
<summary>Using uvx</summary>

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
</details>

<details>
<summary>Using pip</summary>

```json
{
  "mcpServers": {
    "mcsa": {
      "command": "python",
      "args": ["-m", "mcp_server_mcsa"]
    }
  }
}
```
</details>

### VS Code

Add to `.vscode/mcp.json` in your workspace:

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

Or with pip:

```json
{
  "servers": {
    "mcsa": {
      "command": "python",
      "args": ["-m", "mcp_server_mcsa"]
    }
  }
}
```

## Usage Examples

### Real Signal — One-Shot Diagnosis

The fastest way to analyse a measured signal is the `diagnose_from_file`
tool.  Simply provide the file path and motor nameplate data:

> "Diagnose the motor from `C:\data\motor_phaseA.csv` — 50 Hz supply,
>  4 poles, 1470 RPM"

The server loads the file, preprocesses the signal, computes the spectrum,
runs all fault detectors, and returns a complete JSON report with
severity-classified results.

### Step-by-Step Workflow

1. **Load a measured signal** (or generate a synthetic one):
   > "Load the signal from `measurement.wav`"  
   > or: "Generate a test signal with a broken-rotor-bar fault"

2. **Calculate motor parameters**:
   > "Calculate motor parameters for a 4-pole motor, 50 Hz supply, running at 1470 RPM"

3. **Compute expected fault frequencies**:
   > "What are the expected fault frequencies for this motor?"

4. **Analyse the spectrum**:
   > "Compute the FFT spectrum of this signal"

5. **Detect specific faults**:
   > "Check for broken rotor bars in this spectrum"

6. **Envelope analysis (optional)**:
   > "Compute the envelope spectrum to check for bearing modulation"

### Quick Diagnosis from Signal Array

The `run_full_diagnosis` tool runs the entire pipeline on a signal
already in memory in a single call:

```
Input: raw signal array + motor nameplate data
Output: complete report with fault severities and recommendations
```

### Bearing Analysis

For bearing fault analysis, you need the bearing geometry (number of balls,
ball diameter, pitch diameter, contact angle). The server will:
1. Calculate characteristic defect frequencies (BPFO, BPFI, BSF, FTF)
2. Compute expected current sidebands
3. Search the spectrum for those sidebands

### Supported File Formats

| Format | Extensions | Sampling Rate |
|--------|------------|---------------|
| CSV / TSV | `.csv`, `.tsv`, `.txt` | From time column or user-supplied |
| WAV | `.wav` | Embedded in header |
| NumPy | `.npy` | User-supplied |

## Fault Detection Theory

### Broken Rotor Bars (BRB)
Sidebands at $(1 \pm 2s) \cdot f_s$ where $s$ is slip and $f_s$ is supply frequency.
Severity is classified by the dB ratio of sideband to fundamental amplitude.

### Eccentricity
Sidebands at $f_s \pm k \cdot f_r$ where $f_r$ is the rotor mechanical frequency.

### Stator Inter-Turn Faults
Sidebands at $f_s \pm 2k \cdot f_r$ due to winding asymmetry.

### Bearing Defects
Torque oscillations modulate the stator current, creating sidebands at $f_s \pm k \cdot f_{defect}$.
Defect frequencies depend on bearing geometry (BPFO, BPFI, BSF, FTF).

### Severity Thresholds (dB below fundamental)

| Level | Range |
|-------|-------|
| Healthy | ≤ −50 dB |
| Incipient | −50 to −45 dB |
| Moderate | −45 to −40 dB |
| Severe | > −35 dB |

> **Note**: These are general guidelines. Actual thresholds should be adapted to the specific motor, load, and application based on baseline measurements.

## Development

### Setup

```bash
git clone https://github.com/LGDiMaggio/mcp-motor-current-signature-analysis.git
cd mcp-motor-current-signature-analysis
uv sync --dev
```

### Run tests

```bash
uv run pytest
```

### Run with MCP Inspector

```bash
uv run mcp dev src/mcp_server_mcsa/server.py
```

### Lint and type check

```bash
uv run ruff check src/ tests/
uv run pyright src/
```

## Dependencies

- [mcp](https://pypi.org/project/mcp/) — Model Context Protocol SDK
- [numpy](https://numpy.org/) — numerical computing
- [scipy](https://scipy.org/) — signal processing (FFT, filtering, Hilbert transform)
- [pydantic](https://docs.pydantic.dev/) — data validation

## Documentation

For a detailed reference of every tool, resource, and prompt — including
parameter tables, diagnostic workflows, integration patterns, and severity
thresholds — see the **[Usage Guide](USAGE_GUIDE.md)**.

## License

MIT — see [LICENSE](LICENSE) for details.
