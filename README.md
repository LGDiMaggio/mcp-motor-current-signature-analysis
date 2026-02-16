# mcp-server-mcsa

<!-- mcp-name: io.github.LGDiMaggio/mcp-server-mcsa -->

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-compatible-green.svg)](https://modelcontextprotocol.io)

A **Model Context Protocol (MCP) server** for **Motor Current Signature Analysis (MCSA)** — non-invasive spectral analysis and fault detection in electric motors using stator-current signals.

> **mcp-server-mcsa** turns any LLM into a predictive-maintenance expert. By integrating advanced techniques such as **Fast Fourier Transform (FFT)** and **envelope analysis**, the system can *listen* to a motor's electrical signature and automatically identify mechanical and electrical anomalies — all through natural language.

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
- **Persistent data store** — signals and spectra saved to `~/.mcsa_data/` as compressed `.npz` files; referenced by short IDs (`sig_xxxx`, `spec_xxxx`) to keep large arrays out of the chat context; data survives server restarts

## Tools (21)

| Tool | Description |
|------|-------------|
| `inspect_signal_file` | Inspect a signal file format and metadata without loading |
| `load_signal_from_file` | Load a current signal from CSV / WAV / NPY file → returns `signal_id` |
| `calculate_motor_params` | Compute slip, sync speed, rotor frequency from motor data |
| `compute_fault_frequencies` | Calculate expected fault frequencies for all common fault types |
| `compute_bearing_frequencies` | Calculate BPFO, BPFI, BSF, FTF from bearing geometry |
| `preprocess_signal` | DC removal, filtering, normalisation, windowing pipeline → returns new `signal_id` |
| `compute_spectrum` | Single-sided FFT amplitude spectrum → returns `spectrum_id` |
| `compute_power_spectral_density` | Welch PSD estimation → returns `spectrum_id` |
| `find_spectrum_peaks` | Detect and characterise peaks in a spectrum |
| `detect_broken_rotor_bars` | BRB fault index with severity classification |
| `detect_eccentricity` | Air-gap eccentricity detection via sidebands |
| `detect_stator_faults` | Stator inter-turn short circuit detection |
| `detect_bearing_faults` | Bearing defect detection from current spectrum |
| `compute_envelope_spectrum` | Hilbert envelope spectrum for modulation analysis |
| `compute_band_energy` | Integrated spectral energy in a frequency band |
| `compute_time_frequency` | STFT analysis with optional frequency tracking |
| `generate_test_current_signal` | Synthetic motor current with optional faults → returns `signal_id` |
| `run_full_diagnosis` | Complete MCSA diagnostic pipeline from signal or `signal_id` |
| `diagnose_from_file` | Complete MCSA diagnostic pipeline directly from file |
| `list_stored_data` | List all signals and spectra persisted on disk |
| `clear_stored_data` | Delete one or all stored items from disk |

## Resources

| URI | Description |
|-----|-------------|
| `mcsa://fault-signatures` | Reference table of fault signatures, frequencies, and empirical thresholds |

## Prompts

| Prompt | Description |
|--------|-------------|
| `analyze_motor_current` | Step-by-step guided workflow for MCSA analysis |

## Installation & Setup

### Step 1 — Install uv (one-time, if you don't have it)

[uv](https://docs.astral.sh/uv/) is the recommended Python package manager. It handles everything (Python, packages, virtual environments) in a single tool and is used throughout the MCP ecosystem.

**Windows** (PowerShell):
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

> After installing, **restart your terminal** so the `uv` / `uvx` commands are available.

### Step 2 — Verify it works

```bash
uvx mcp-server-mcsa --help
```

You should see the help text. **That's it** — no `pip install` needed. `uvx` downloads and runs the package automatically in an isolated environment.

### Step 3 — Add to your MCP client

Pick your client and add the configuration below. **No other steps are required.**

#### Claude Desktop

Open the config file:
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

Add `mcsa` inside the `mcpServers` object (create the file if it doesn't exist):

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

Then **restart Claude Desktop**.

#### VS Code (Copilot / Continue)

Create (or edit) `.vscode/mcp.json` in your workspace:

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

#### Cursor

Go to **Settings → MCP Servers → Add new server**:
- Type: `command`
- Command: `uvx mcp-server-mcsa`

### Step 4 — Test

In your MCP client, try:

> "Generate a test signal with a broken rotor bar fault and run a full diagnosis. Motor: 4 poles, 50 Hz, 1470 RPM."

If the server responds with a diagnostic report, you're all set.

---

<details>
<summary><strong>Alternative: install with pip</strong> (not recommended — see note)</summary>

```bash
pip install mcp-server-mcsa
```

Then configure your client with:

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

> **⚠️ Common issue on Windows**: if you installed Python from the Microsoft Store, the `mcp-server-mcsa` command may not be in your PATH, causing a "server disconnected" error. In that case, find your Python path with `python -c "import sys; print(sys.executable)"` and use the full path in the config:
>
> ```json
> {
>   "mcpServers": {
>     "mcsa": {
>       "command": "C:/Users/YOU/AppData/Local/.../python.exe",
>       "args": ["-m", "mcp_server_mcsa"]
>     }
>   }
> }
> ```
>
> Using `uvx` avoids this problem entirely.

</details>

<details>
<summary><strong>Alternative: install from source</strong> (for development)</summary>

```bash
git clone https://github.com/LGDiMaggio/mcp-motor-current-signature-analysis.git
cd mcp-motor-current-signature-analysis
uv sync --dev
```

Configure the client to point to the local repo:

```json
{
  "mcpServers": {
    "mcsa": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/mcp-motor-current-signature-analysis", "run", "mcp-server-mcsa"]
    }
  }
}
```

Run tests:

```bash
uv run pytest
```

Debug with MCP Inspector:

```bash
uv run mcp dev src/mcp_server_mcsa/server.py
```

</details>

### Troubleshooting

| Problem | Fix |
|---------|-----|
| "server disconnected" on Claude Desktop | Check the logs at `%APPDATA%\Claude\logs\` (Windows) or `~/Library/Logs/Claude/` (macOS). Most common cause: the command in the config is not found. Use `uvx` to avoid PATH issues. |
| `uvx: command not found` | Restart your terminal after installing uv. On Windows, you may need to close and reopen PowerShell. |
| `mcp-server-mcsa: command not found` (pip) | The script wasn't added to PATH. Use `python -m mcp_server_mcsa` instead, or switch to `uvx`. |
| Server starts but tools don't appear | Make sure you restarted the MCP client after editing the config. |

## Data Store

Signals and spectra are **persisted to disk** as compressed `.npz` files
in `~/.mcsa_data/` (configurable via the `MCSA_DATA_DIR` environment
variable).  This means:

- **Large arrays never enter the chat** — only short IDs (`sig_xxxx`,
  `spec_xxxx`) and compact summaries are returned to the LLM.
- **Data survives server restarts** — reopen Claude Desktop tomorrow and
  your signals are still there.
- **All data in one place** — loaded measurements and generated test
  signals live side by side in the same folder.

```
~/.mcsa_data/
  signals/
    sig_a1b2c3d4.npz   ← loaded from CSV
    sig_e5f6g7h8.npz   ← generated test signal
  spectra/
    spec_i9j0k1l2.npz  ← FFT result
```

Use `list_stored_data` to see everything on disk and `clear_stored_data`
to remove items.

## Usage Examples

### Real Signal — One-Shot Diagnosis

The fastest way to analyse a measured signal is the `diagnose_from_file`
tool.  Simply provide the file path and motor nameplate data:

> "Diagnose the motor from `C:\data\motor_phaseA.csv` — 50 Hz supply,
>  4 poles, 1470 RPM"

The server loads the file, preprocesses the signal, computes the spectrum,
runs all fault detectors, and returns a complete JSON report with
severity-classified results.

### Step-by-Step Workflow (with signal IDs)

1. **Load a measured signal** (or generate a synthetic one):
   > "Load the signal from `measurement.wav`" → returns `signal_id: sig_a1b2`
   > or: "Generate a test signal with a broken-rotor-bar fault" → `sig_c3d4`

2. **Calculate motor parameters**:
   > "Calculate motor parameters for a 4-pole motor, 50 Hz supply, running at 1470 RPM"

3. **Compute expected fault frequencies**:
   > "What are the expected fault frequencies for this motor?"

4. **Preprocess the signal**:
   > "Preprocess signal sig_a1b2" → returns new `signal_id: sig_e5f6`

5. **Analyse the spectrum**:
   > "Compute the FFT spectrum of sig_e5f6" → returns `spectrum_id: spec_g7h8`

6. **Detect specific faults**:
   > "Check for broken rotor bars in spec_g7h8"

7. **Envelope analysis (optional)**:
   > "Compute the envelope spectrum of sig_e5f6"

### Quick Diagnosis from Stored Signal

The `run_full_diagnosis` tool runs the entire pipeline on a stored signal
in a single call:

```
Input: signal_id + motor nameplate data
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

## Citation

If you use this software in your research, please cite it:

```bibtex
@software{dimaggio_mcsa_2025,
  author       = {Di Maggio, Luigi Gianpio},
  title        = {mcp-server-mcsa: MCP Server for Motor Current Signature Analysis},
  year         = 2025,
  url          = {https://github.com/LGDiMaggio/mcp-motor-current-signature-analysis},
  license      = {MIT}
}
```

> GitHub shows a **"Cite this repository"** button automatically from the [`CITATION.cff`](CITATION.cff) file.

[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--2295--8944-green.svg)](https://orcid.org/0000-0002-2295-8944)

## License

MIT — see [LICENSE](LICENSE) for details.
