#!/usr/bin/env python3
"""
SynAgent Round-Trip Validation Pipeline
Monte Carlo self-consistency checking for SynLlama retrosynthesis outputs.

Usage:
    python synagent_pipeline.py synllama-raw-output.csv
    python synagent_pipeline.py synllama-raw-output.csv --n-samples 100 --threshold 0.6
    python synagent_pipeline.py synllama-raw-output.csv --backend local
    python synagent_pipeline.py synllama-raw-output.csv --backend savio

Backends:
    mock   — simulated SynLlama (for development/testing, no model required)
    local  — SynLlama running locally on HTTP API (e.g., localhost:8000)
    savio  — SynLlama running on Savio/NERSC (wire in your endpoint below)
"""

import asyncio
import json
import re
import csv
import argparse
import sys
import random
import math
from pathlib import Path
from typing import Optional, Callable, Awaitable
from datetime import datetime

# ── Install check ────────────────────────────────────────────────────────────
try:
    from pydantic import BaseModel
except ImportError:
    sys.exit("Missing dependency: pip install pydantic rdkit")

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("[WARN] RDKit not available — RDKit template checks will be skipped")


# ═════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═════════════════════════════════════════════════════════════════════════════

class ReactionStep(BaseModel):
    step_index: int
    product: str
    reactants: list[str]
    reaction_template: Optional[str] = None
    sampling_param: Optional[str] = None
    # Populated after validation:
    hit_count: int = 0
    n_samples: int = 0
    confidence: float = 0.0
    flagged: bool = False
    rdkit_valid: bool = False
    rdkit_note: str = ""

class ValidationResult(BaseModel):
    molecule_smiles: str
    steps: list[ReactionStep]
    overall_confidence: float = 0.0
    passed: bool = False
    threshold: float = 0.5
    n_samples_used: int = 0


# ═════════════════════════════════════════════════════════════════════════════
# SMILES UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def canonicalize(smiles: str) -> Optional[str]:
    if not RDKIT_AVAILABLE:
        return smiles.strip()
    mol = Chem.MolFromSmiles(smiles.strip())
    return Chem.MolToSmiles(mol) if mol else None

def smiles_match(s1: str, s2: str) -> bool:
    c1, c2 = canonicalize(s1), canonicalize(s2)
    return bool(c1 and c2 and c1 == c2)


# ═════════════════════════════════════════════════════════════════════════════
# RDKIT VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def rdkit_validate_step(step: ReactionStep) -> tuple[bool, str]:
    """
    Forward-execute reaction template on reactants.
    Returns (is_valid, note).
    """
    if not RDKIT_AVAILABLE:
        return True, "RDKit not available — skipped"

    # Check reactant + product SMILES validity
    for r in step.reactants:
        if not canonicalize(r):
            return False, f"Invalid reactant SMILES: {r}"
    if not canonicalize(step.product):
        return False, f"Invalid product SMILES: {step.product}"

    if not step.reaction_template:
        return True, "No template — SMILES validity OK"

    try:
        template = re.sub(r'</?rxn>', '', step.reaction_template).strip()
        rxn = AllChem.ReactionFromSmarts(template)
        if rxn is None:
            return False, "Could not parse reaction template"

        reactant_mols = [Chem.MolFromSmiles(r) for r in step.reactants]
        if any(m is None for m in reactant_mols):
            return False, "Failed to parse one or more reactant SMILES"

        products = rxn.RunReactants(tuple(reactant_mols))
        if not products:
            return False, "Template produced no products on these reactants"

        target = canonicalize(step.product)
        for product_set in products:
            for prod in product_set:
                try:
                    if Chem.MolToSmiles(prod) == target:
                        return True, "Template forward-execution matched product"
                except Exception:
                    continue
        return False, "Template ran but product not recovered"

    except Exception as e:
        return False, f"RDKit exception: {e}"


# ═════════════════════════════════════════════════════════════════════════════
# SYNLLAMA BACKENDS
# ═════════════════════════════════════════════════════════════════════════════

# Type alias
SynLlamaFn = Callable[[list[str], str], Awaitable[Optional[str]]]


async def mock_synllama(reactants: list[str], sampling_param: str = "medium") -> Optional[str]:
    """
    Mock SynLlama for development.
    Simulates ~65% hit rate with some variance by sampling param.
    Replace this with local_synllama or savio_synllama for real runs.
    """
    hit_rates = {"frozen": 0.80, "low": 0.70, "medium": 0.65, "high": 0.55}
    hit_rate = hit_rates.get(sampling_param, 0.65)
    await asyncio.sleep(0.005)  # simulate tiny latency
    # Returns None (miss) — a real backend would return the predicted product SMILES
    # We use None as "miss" since we don't have the actual product to return
    return None  # mock always returns None; real fn returns predicted SMILES


async def mock_synllama_with_product(
    reactants: list[str],
    sampling_param: str,
    true_product: str,
    hit_rate: float
) -> Optional[str]:
    """
    Enhanced mock: actually returns the product on hits.
    Used internally for demo runs — replace with real SynLlama.
    """
    await asyncio.sleep(0.005)
    return true_product if random.random() < hit_rate else None


async def local_synllama(reactants: list[str], sampling_param: str = "medium") -> Optional[str]:
    """
    Local SynLlama via HTTP API.

    Wire up: start SynLlama locally, expose it on an HTTP endpoint, then
    adjust the URL and payload format below to match your server.

    Example server (FastAPI sketch):
        POST /retro  →  {"smiles": "reactant1.reactant2", "sampling": "medium"}
        Response:       {"product": "SMILES_STRING"}
    """
    try:
        import aiohttp
    except ImportError:
        raise RuntimeError("pip install aiohttp to use the local backend")

    url = "http://localhost:8000/retro"  # ← adjust to your local server
    payload = {
        "smiles": ".".join(reactants),
        "sampling": sampling_param,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("product")
            return None


async def savio_synllama(reactants: list[str], sampling_param: str = "medium") -> Optional[str]:
    """
    Savio/NERSC SynLlama backend.

    Two patterns depending on your setup:
    (A) REST endpoint: if SynLlama is running as a persistent service on a login node
        → same pattern as local_synllama but with Savio URL + auth token
    (B) Batch job: submit a SLURM job, poll for completion, read result
        → use paramiko or the Savio REST API

    Wire-in checklist:
        1. Set SAVIO_URL and SAVIO_TOKEN env vars (or hardcode below)
        2. Match payload format to your SynLlama server's API
        3. Handle SLURM queue latency if using batch mode
    """
    import os
    try:
        import aiohttp
    except ImportError:
        raise RuntimeError("pip install aiohttp to use the savio backend")

    url = os.environ.get("SAVIO_SYNLLAMA_URL", "https://savio.lbl.gov/synllama/retro")  # ← adjust
    token = os.environ.get("SAVIO_SYNLLAMA_TOKEN", "")

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    payload = {"smiles": ".".join(reactants), "sampling": sampling_param}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers,
                                timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("product")
            return None


# ═════════════════════════════════════════════════════════════════════════════
# MONTE CARLO ROUND-TRIP VALIDATOR
# ═════════════════════════════════════════════════════════════════════════════

class RoundTripValidator:
    """
    Core Monte Carlo validator.

    Algorithm:
        For each retrosynthetic step (product → reactants):
            1. Run RDKit template check
            2. Feed reactants back into SynLlama N times (async, parallelized)
            3. Count how many runs recover the original product
            4. confidence = hit_count / N
            5. Flag step if confidence < threshold

    The stochastic sampling exploits SynLlama's temperature-based generation:
    a chemically unambiguous reaction should consistently produce the same
    product from its reactants; an ambiguous one will scatter across products.
    """

    def __init__(
        self,
        n_samples: int = 50,
        threshold: float = 0.5,
        synllama_fn: Optional[SynLlamaFn] = None,
        demo_mode: bool = False,
    ):
        self.n_samples = n_samples
        self.threshold = threshold
        self.synllama_fn = synllama_fn or mock_synllama
        self.demo_mode = demo_mode  # uses enhanced mock that actually returns products

    async def _single_sample(self, step: ReactionStep) -> Optional[str]:
        """One stochastic SynLlama call."""
        if self.demo_mode:
            # Demo: simulate realistic hit rates per sampling param
            rates = {"frozen": 0.82, "low": 0.68, "medium": 0.63, "high": 0.50}
            rate = rates.get(step.sampling_param or "medium", 0.63)
            return await mock_synllama_with_product(
                step.reactants, step.sampling_param or "medium",
                step.product, rate
            )
        return await self.synllama_fn(step.reactants, step.sampling_param or "medium")

    async def validate_step(self, step: ReactionStep) -> ReactionStep:
        """
        Validate one retrosynthetic step:
        - RDKit template forward check
        - Monte Carlo N-sample round-trip
        """
        # 1. RDKit check
        rdkit_ok, rdkit_note = rdkit_validate_step(step)
        step.rdkit_valid = rdkit_ok
        step.rdkit_note = rdkit_note

        # 2. Monte Carlo round-trip: N samples in parallel
        tasks = [self._single_sample(step) for _ in range(self.n_samples)]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        target = canonicalize(step.product)
        hit_count = 0
        for result in raw_results:
            if isinstance(result, Exception) or result is None:
                continue
            predicted = canonicalize(result)
            if predicted and predicted == target:
                hit_count += 1

        step.hit_count = hit_count
        step.n_samples = self.n_samples
        step.confidence = hit_count / self.n_samples if self.n_samples > 0 else 0.0
        step.flagged = step.confidence < self.threshold

        return step

    async def validate_pathway(
        self, steps: list[ReactionStep], molecule_smiles: str
    ) -> ValidationResult:
        """Validate all steps in a retrosynthetic pathway sequentially."""
        validated = []
        for step in steps:
            validated.append(await self.validate_step(step))

        # Overall confidence = min across steps (conservative)
        overall = min((s.confidence for s in validated), default=0.0)

        return ValidationResult(
            molecule_smiles=molecule_smiles,
            steps=validated,
            overall_confidence=overall,
            passed=overall >= self.threshold,
            threshold=self.threshold,
            n_samples_used=self.n_samples,
        )


# ═════════════════════════════════════════════════════════════════════════════
# CSV PARSER  (synllama-raw-output.csv format)
# ═════════════════════════════════════════════════════════════════════════════

def _strip_tag(text: str, tag: str) -> str:
    return re.sub(rf'</?{re.escape(tag)}>', '', text).strip()


def parse_synllama_csv(csv_path: str) -> list[tuple[str, list[ReactionStep]]]:
    """
    Parse synllama-raw-output.csv.

    Expected columns: smiles, sampling_params, response
    response is a JSON blob with "reactions" and "building_blocks".

    Returns: list of (target_smiles, [ReactionStep, ...])
    """
    results = []
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            target_smiles = row.get('smiles', '').strip()
            sampling_param = row.get('sampling_params', 'medium').strip()
            response_str = row.get('response', '').strip()

            if not response_str:
                print(f"  [WARN] Row {row_idx+1}: empty response, skipping")
                continue

            try:
                response = json.loads(response_str)
            except json.JSONDecodeError as e:
                print(f"  [WARN] Row {row_idx+1}: JSON parse error ({e}), skipping")
                continue

            steps: list[ReactionStep] = []
            for rxn in response.get('reactions', []):
                product_raw = rxn.get('product', '').strip()
                reactants_raw = rxn.get('reactants', [])
                template_raw = rxn.get('reaction_template', '')

                product = canonicalize(product_raw) or product_raw
                reactants = [canonicalize(r) or r for r in reactants_raw]
                template = _strip_tag(template_raw, 'rxn') if template_raw else None

                steps.append(ReactionStep(
                    step_index=rxn.get('reaction_number', len(steps) + 1),
                    product=product,
                    reactants=reactants,
                    reaction_template=template,
                    sampling_param=sampling_param,
                ))

            if steps:
                results.append((target_smiles, steps))

    return results


# ═════════════════════════════════════════════════════════════════════════════
# REPORT GENERATOR
# ═════════════════════════════════════════════════════════════════════════════

def _bar(value: float, width: int = 18) -> str:
    filled = round(value * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {value:5.1%}"


def generate_report(results: list[ValidationResult], output_path: Optional[str] = None) -> str:
    if not results:
        return "No results to report."

    threshold = results[0].threshold
    n = results[0].n_samples_used
    passed = [r for r in results if r.passed]
    flagged = [r for r in results if not r.passed]

    lines = [
        "=" * 72,
        "  SynAgent — Round-Trip Monte Carlo Validation Report",
        f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  N samples : {n}  |  Threshold : {threshold}  |  Molecules : {len(results)}",
        "=" * 72,
        "",
        f"  PASSED  : {len(passed):3d} / {len(results)}",
        f"  FLAGGED : {len(flagged):3d} / {len(results)}",
        "",
    ]

    for i, result in enumerate(results, 1):
        status = "✓ PASS" if result.passed else "✗ FLAGGED"
        smiles_display = (result.molecule_smiles[:64] + "…") if len(result.molecule_smiles) > 64 else result.molecule_smiles
        lines += [
            "─" * 72,
            f"[{i:02d}] {smiles_display}",
            f"     Overall  {_bar(result.overall_confidence)}  {status}",
        ]
        for step in result.steps:
            rdkit_icon = "✓" if step.rdkit_valid else "✗"
            flag_note = "  ← FLAGGED" if step.flagged else ""
            lines.append(
                f"     Step {step.step_index}   {_bar(step.confidence, 14)}  "
                f"RDKit {rdkit_icon}  MC {step.hit_count:3d}/{step.n_samples}{flag_note}"
            )
            if step.rdkit_note and not step.rdkit_valid:
                lines.append(f"              ↳ {step.rdkit_note}")

    lines += ["", "=" * 72, ""]

    # Flagged summary table
    if flagged:
        lines += ["  FLAGGED STEPS — review or escalate to ASKCOS", ""]
        lines.append(f"  {'#':<4} {'SMILES (truncated)':<45} {'Conf':>6}  {'MC hits':>8}")
        lines.append("  " + "─" * 68)
        for result in flagged:
            for step in result.steps:
                if step.flagged:
                    sm = result.molecule_smiles[:44]
                    lines.append(
                        f"  {step.step_index:<4} {sm:<45} {step.confidence:>5.1%}  {step.hit_count:>3}/{step.n_samples:<3}"
                    )
        lines.append("")

    report = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(report, encoding='utf-8')

    return report


def export_json(results: list[ValidationResult], output_path: str):
    data = [r.model_dump() for r in results]
    Path(output_path).write_text(json.dumps(data, indent=2), encoding='utf-8')


def export_csv(results: list[ValidationResult], output_path: str):
    """
    Export one row per reaction step — clean format for PI review or Excel.

    Columns:
        molecule_smiles     — target molecule being synthesised
        sampling_param      — frozen / low / medium / high
        step_index          — reaction number within the pathway
        reactants           — reactant SMILES joined by ' + '
        product             — product SMILES
        reaction_type       — short human-readable label inferred from template
        rdkit_valid         — TRUE / FALSE (forward template check)
        mc_hits             — e.g. 44 (raw hit count)
        n_samples           — e.g. 50
        confidence          — 0.00–1.00
        confidence_pct      — e.g. 88.0%
        status              — PASS / FLAGGED
        pathway_status      — PASS / FLAGGED (overall for the molecule)
        threshold           — confidence threshold used
        notes               — plain-English summary for PI
    """
    import csv as csv_mod

    def infer_reaction_type(template: Optional[str]) -> str:
        if not template:
            return "unknown"
        t = template.lower()
        if "n+0" in t and "c:2" in t and "o:4" in t:
            return "amide coupling"
        if "c1ccccc1" in t or "aromatic" in t:
            return "aromatic substitution"
        if "cl,f" in t and "[n;" in t:
            return "SNAr (C-N coupling)"
        if "br" in t or "i]" in t:
            return "cross-coupling"
        if "o-h" in t or "c-oh" in t:
            return "esterification"
        return "unclassified"

    fieldnames = [
        "molecule_smiles", "sampling_param", "step_index",
        "reactants", "product",
        "reaction_type", "rdkit_valid",
        "mc_hits", "n_samples", "confidence", "confidence_pct",
        "status", "pathway_status", "threshold", "notes",
    ]

    rows = []
    for result in results:
        pathway_status = "PASS" if result.passed else "FLAGGED"
        for step in result.steps:
            status = "PASS" if not step.flagged else "FLAGGED"

            # Plain-English note for PI
            if not step.rdkit_valid:
                note = f"RDKit template check failed: {step.rdkit_note}"
            elif step.flagged:
                note = (
                    f"Low self-consistency: only {step.hit_count}/{step.n_samples} "
                    f"MC samples recovered product — reactants may be chemically ambiguous"
                )
            else:
                note = (
                    f"Self-consistent: {step.hit_count}/{step.n_samples} samples "
                    f"recovered product ({step.confidence:.0%} confidence)"
                )

            rows.append({
                "molecule_smiles":  result.molecule_smiles,
                "sampling_param":   step.sampling_param or "",
                "step_index":       step.step_index,
                "reactants":        " + ".join(step.reactants),
                "product":          step.product,
                "reaction_type":    infer_reaction_type(step.reaction_template),
                "rdkit_valid":      "TRUE" if step.rdkit_valid else "FALSE",
                "mc_hits":          step.hit_count,
                "n_samples":        step.n_samples,
                "confidence":       round(step.confidence, 4),
                "confidence_pct":   f"{step.confidence:.1%}",
                "status":           status,
                "pathway_status":   pathway_status,
                "threshold":        result.threshold,
                "notes":            note,
            })

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

async def run_pipeline(
    csv_path: str,
    n_samples: int = 50,
    threshold: float = 0.5,
    synllama_fn: Optional[SynLlamaFn] = None,
    demo_mode: bool = False,
    output_dir: str = ".",
    verbose: bool = True,
    write_csv: bool = True,
) -> list[ValidationResult]:

    print(f"\n{'═' * 60}")
    print("  SynAgent Round-Trip Validation Pipeline")
    print(f"{'═' * 60}")
    print(f"  Input      : {csv_path}")
    print(f"  N samples  : {n_samples}  (Monte Carlo per step)")
    print(f"  Threshold  : {threshold}")
    print(f"  Backend    : {'demo mock' if demo_mode else 'custom / mock'}")
    print(f"  Output dir : {output_dir}")
    print()

    # Parse CSV
    print("[1/3] Parsing input CSV...")
    parsed = parse_synllama_csv(csv_path)
    print(f"       → {len(parsed)} molecules loaded")
    total_steps = sum(len(steps) for _, steps in parsed)
    print(f"       → {total_steps} reaction steps total")
    print(f"       → {total_steps * n_samples} SynLlama calls queued\n")

    # Validate
    print("[2/3] Running Monte Carlo validation...")
    validator = RoundTripValidator(
        n_samples=n_samples,
        threshold=threshold,
        synllama_fn=synllama_fn,
        demo_mode=demo_mode,
    )

    all_results: list[ValidationResult] = []
    for idx, (target_smiles, steps) in enumerate(parsed, 1):
        short = target_smiles[:48] + ("…" if len(target_smiles) > 48 else "")
        print(f"  [{idx:02d}/{len(parsed):02d}] {short}", end="", flush=True)
        result = await validator.validate_pathway(steps, target_smiles)
        all_results.append(result)
        status = "PASS" if result.passed else "FLAGGED"
        print(f"  →  {result.overall_confidence:.1%}  [{status}]")

    # Report
    print(f"\n[3/3] Generating reports...")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = str(out / "validation_report.txt")
    json_path = str(out / "validation_results.json")

    report = generate_report(all_results, output_path=report_path)
    export_json(all_results, json_path)

    if write_csv:
        csv_out_path = str(out / "validation_results.csv")
        n_rows = export_csv(all_results, csv_out_path)
        print(f"  CSV    : {csv_out_path}  ({n_rows} reaction rows)")

    if verbose:
        print()
        print(report)

    print(f"  Report : {report_path}")
    print(f"  JSON   : {json_path}\n")

    return all_results


# ═════════════════════════════════════════════════════════════════════════════
# CLI ENTRYPOINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SynAgent Monte Carlo Round-Trip Validation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick demo run with mock SynLlama (no model needed):
  python synagent_pipeline.py synllama-raw-output.csv --demo

  # Standard pipeline run (N=50, real local model):
  python synagent_pipeline.py synllama-raw-output.csv --backend local

  # High-confidence run for paper/PI review:
  python synagent_pipeline.py synllama-raw-output.csv --backend savio --n-samples 100

  # Strict threshold:
  python synagent_pipeline.py synllama-raw-output.csv --threshold 0.7 --n-samples 50
        """,
    )
    parser.add_argument("csv", help="Path to synllama-raw-output.csv")
    parser.add_argument(
        "--n-samples", type=int, default=50,
        help="Monte Carlo samples per reaction step (default: 50)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Confidence threshold below which steps are flagged (default: 0.5)"
    )
    parser.add_argument(
        "--backend", choices=["mock", "local", "savio"], default="mock",
        help="SynLlama inference backend (default: mock)"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run with demo mock that simulates realistic hit rates (no model needed)"
    )
    parser.add_argument(
        "--output-dir", default=".", metavar="DIR",
        help="Directory for validation_report.txt and validation_results.json"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress full report print; only save to files"
    )
    parser.add_argument(
        "--no-csv", action="store_true",
        help="Skip CSV output (only write .txt report and .json)"
    )
    args = parser.parse_args()

    backend_map: dict[str, SynLlamaFn] = {
        "mock": mock_synllama,
        "local": local_synllama,
        "savio": savio_synllama,
    }

    asyncio.run(run_pipeline(
        csv_path=args.csv,
        n_samples=args.n_samples,
        threshold=args.threshold,
        synllama_fn=backend_map[args.backend],
        demo_mode=args.demo,
        output_dir=args.output_dir,
        verbose=not args.quiet,
        write_csv=not args.no_csv,
    ))


if __name__ == "__main__":
    main()
