# SynAgent Pipeline — AI Handoff Prompt
# Copy everything below this line into a new AI session

---

You are helping me set up and run an automated retrosynthesis validation pipeline called SynAgent. Here is the full context of what has been built, what exists, and exactly what needs to be done next.

---

## What This Project Is

SynLlama (from the Head-Gordon Lab at LBNL, https://github.com/THGLab/SynLlama) is an LLM fine-tuned on retrosynthesis. It takes a target molecule (SMILES string) and predicts what reactants it can be made from.

The pipeline I've built validates SynLlama's outputs using a Monte Carlo round-trip consistency check:
1. SynLlama generates: product → [reactant_1, reactant_2]
2. Feed reactants back into SynLlama N=50 times (stochastic sampling)
3. Count how many runs recover the original product
4. confidence = hit_count / N
5. If confidence < 0.5 → flag the step for review

This is cheap ($0.001/call × 50 = $0.05/step) compared to external validators like ASKCOS, so it acts as a pre-filter.

---

## Files In This Project Folder

### synagent_pipeline.py
The full automated pipeline. Key components:
- `parse_synllama_csv()` — parses SynLlama's raw CSV output format
- `rdkit_validate_step()` — RDKit forward template execution check (fast, rule-based)
- `RoundTripValidator` — async Monte Carlo sampler, runs N SynLlama calls in parallel
- `generate_report()` — human-readable .txt report with confidence bars
- `export_csv()` — per-reaction CSV with 15 columns for PI review
- `export_json()` — machine-readable full results

Three SynLlama backends are already stubbed in:
- `mock_synllama` — returns None (placeholder, no model needed)
- `local_synllama` — HTTP POST to localhost:8000/retro (fill in when running locally)
- `savio_synllama` — NERSC/Savio endpoint (fill in for batch runs)

### synllama-sample.csv
Test data with 9 rows: same molecule (Cc1csc(NC(=O)c2cccnc2N2CCOCC2)n1) at different
sampling params (frozen/low/medium/high) including one SNAr reaction with different reactants.
Input format: smiles, sampling_params, response (JSON blob with reactions + building_blocks).

### validation_results.csv
Output from last demo run — one row per reaction step with columns:
molecule_smiles, sampling_param, step_index, reactants, product, reaction_type,
rdkit_valid, mc_hits, n_samples, confidence, confidence_pct, status, pathway_status,
threshold, notes

### validation_report.txt
Human-readable text report from last demo run.

---

## Hardware

- **Laptop (use this for SynLlama):** NVIDIA RTX 3060 GPU — run SynLlama here with CUDA
- **MacBook Air M4 16GB (macOS Sequoia 15.7.7)** — run the pipeline from here if needed,
  but SynLlama inference should stay on the 3060

---

## What Needs To Be Done — In Order

### STEP 1: Install SynLlama on the 3060 laptop

```bash
git clone https://github.com/THGLab/SynLlama
cd SynLlama
conda env create -f environment.yml
conda activate synllama
pip install -e .
```

Before running `conda env create`, READ environment.yml first. If it contains
`pytorch-cuda`, `cudatoolkit`, or `nvidia` packages, those are correct for the 3060
(CUDA 11.x or 12.x — match to what `nvidia-smi` shows on that machine).

### STEP 2: Download SynLlama model weights

In the README at https://github.com/THGLab/SynLlama, there is a "here" link for
downloading trained model files. Download them and note the path — you'll need it.
The weights should NOT go in git (they're multi-GB). Store them somewhere stable
on the laptop, e.g. ~/models/synllama/.

### STEP 3: Run a test SynLlama inference manually

Follow the Inference Guide linked in the README to verify SynLlama works standalone
before connecting it to this pipeline. You should be able to pass in a SMILES string
and get back a JSON with reactions + building_blocks — same format as synllama-sample.csv.

### STEP 4: Expose SynLlama as a local HTTP endpoint

The pipeline calls SynLlama via `local_synllama()` in synagent_pipeline.py, which
does a POST to http://localhost:8000/retro. You need to wrap SynLlama inference in
a simple FastAPI server. Here is the skeleton to implement:

```python
# synllama_server.py — run this on the 3060 laptop
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class InferenceRequest(BaseModel):
    smiles: str           # reactant SMILES joined by "."
    sampling: str = "medium"

@app.post("/retro")
async def retro(req: InferenceRequest):
    # TODO: call your SynLlama inference function here
    # result should be the predicted product SMILES string
    product_smiles = YOUR_SYNLLAMA_INFERENCE_FN(req.smiles, req.sampling)
    return {"product": product_smiles}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Install: `pip install fastapi uvicorn`
Run: `python synllama_server.py`

### STEP 5: Wire local_synllama into the pipeline

In synagent_pipeline.py, `local_synllama()` already exists and posts to localhost:8000.
The payload and response format match the server skeleton above — no changes needed
unless your SynLlama server uses different field names.

Test it:
```bash
python synagent_pipeline.py synllama-sample.csv --backend local --n-samples 10
```

### STEP 6: Full pipeline run

```bash
# Standard run (N=50, real SynLlama, outputs to results/ folder)
python synagent_pipeline.py synllama-raw-output.csv --backend local --n-samples 50 --output-dir results/

# High-confidence run for PI/paper
python synagent_pipeline.py synllama-raw-output.csv --backend local --n-samples 100 --output-dir results_paper/
```

Outputs written automatically:
- results/validation_report.txt   — human-readable with confidence bars
- results/validation_results.csv  — per-reaction table, open in Excel for PI review
- results/validation_results.json — machine-readable full results

---

## Running the Pipeline (Quick Reference)

```bash
# Demo (no SynLlama needed, simulated hit rates):
python synagent_pipeline.py synllama-sample.csv --demo --n-samples 50

# Real local SynLlama:
python synagent_pipeline.py your_data.csv --backend local --n-samples 50

# Strict threshold:
python synagent_pipeline.py your_data.csv --backend local --n-samples 50 --threshold 0.7

# Quiet mode (don't print report, just save files):
python synagent_pipeline.py your_data.csv --backend local --n-samples 50 --quiet
```

---

## Input CSV Format

The pipeline expects synllama-raw-output.csv format:

```
smiles,sampling_params,response
PRODUCT_SMILES,medium,"{""reactions"": [{""reaction_number"": 1, ""reaction_template"": ""<rxn>TEMPLATE</rxn>"", ""reactants"": [""SMILES1"", ""SMILES2""], ""product"": ""PRODUCT_SMILES""}], ""building_blocks"": [""<bb>SMILES1</bb>"", ""<bb>SMILES2</bb>""]}"
```

One row per SynLlama sample. Multiple rows with the same molecule at different
sampling_params are fine — each is validated independently.

---

## Key Design Decisions Already Made

- Overall pathway confidence = min across steps (conservative — one weak step flags whole pathway)
- Default threshold = 0.5 (tune empirically by comparing valid vs invalid molecules)
- N=50 for standard runs, N=100 for paper/PI results
- Flagged steps: report only (no auto-ASKCOS call — add later if budget allows)
- The `--demo` flag simulates realistic hit rates per sampling param for testing without a model

---

## Dependencies

```bash
pip install pydantic rdkit aiohttp fastapi uvicorn
```

Or via conda:
```bash
conda install -c conda-forge rdkit
pip install pydantic aiohttp fastapi uvicorn
```

---

## What The Monte Carlo Confidence Scores Mean

| Confidence | Interpretation |
|---|---|
| >0.8 | Strong, unambiguous reaction — high confidence |
| 0.6–0.8 | Good self-consistency — acceptable |
| 0.5–0.6 | Borderline — worth a closer look |
| <0.5 | FLAGGED — reactants are chemically ambiguous or step is weak |

The scores reflect SynLlama's *internal* self-consistency, not ground truth.
A high score means the model agrees with itself. You still need RDKit (already built in)
and optionally ASKCOS for independent external validation.

---

## Next Development Steps (not yet built)

1. ASKCOS forward prediction API call for flagged steps (third validation tier)
2. Add `synllama_server.py` to the repo once SynLlama inference is working
3. Run on synllama-raw-valid.csv vs synllama-raw-output.csv to get baseline
   confidence distributions and tune the 0.5 threshold empirically
4. Add to validation.py in SynAgent as @agent.tool_plain (see original design doc PDF)
