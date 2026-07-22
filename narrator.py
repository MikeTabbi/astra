"""Narration and self-consistency verification for Project ASTRA (Stage 3).

Objectives:
  1. Pydantic models force LLM string output into parseable JSON.
  2. LAN connection to the host MacBook's Ollama node (via inference.py).
  3. 5x consecutive verification loop with variance scoring + adversarial
     contradiction detection, yielding a dashboard reliability rating.

Gene-agnostic: target_gene flows through as a parameter. C5, C3, and Actin
are processed identically with no code changes (spec section 3, Stage 1).
"""

import difflib
import json
import logging
import statistics
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from inference import InferenceProvider, get_provider

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONSISTENCY_RUNS = 5
MAX_ATTEMPTS = 2
PROMPT_CHAINS_PATH = Path("prompt_chains.json")

# Unified table schema boundary (spec Stage 1). All modules agree on this.
SCHEMA_COLUMNS = ["target_name", "time_point", "salinity", "fold_change_rq", "variance_sd"]
SALINITY_LEVELS = ["Low", "Med", "High"]


class NarrativeSummary(BaseModel):
    """Structured narration of one expression matrix."""

    headline: str = Field(description="One-sentence biological takeaway.")
    trend_description: str = Field(description="How expression shifts across time and salinity.")
    peak_condition: str = Field(description="Time point and salinity level of strongest response.")
    peak_time_point: int = Field(ge=1, le=7, description="Time point of peak fold change.")
    peak_salinity: str = Field(description="Salinity level of peak fold change (Low, Med, or High).")
    direction: str = Field(description="One of: upregulation, transcriptional collapse, mixed, flat.")
    confidence_note: str = Field(description="Caveats given technical replicate variance.")
    key_observations: list[str] = Field(min_length=1, max_length=5)

    def as_text(self) -> str:
        """Flatten to a single string for lexical variance comparison."""
        return " ".join([
            self.headline,
            self.trend_description,
            self.peak_condition,
            self.direction,
            self.confidence_note,
            *self.key_observations,
        ])

    def claims(self) -> dict:
        """The factual assertions worth checking for cross-run drift."""
        return {
            "peak_time_point": self.peak_time_point,
            "peak_salinity": self.peak_salinity,
            "direction": self.direction.strip().lower(),
        }


class ContradictionReport(BaseModel):
    """Adversarial evaluation of N narratives over identical data (spec Stage 3)."""

    contradictions_found: bool
    contradiction_details: list[str] = Field(default_factory=list, max_length=10)
    numerical_disagreements: list[str] = Field(default_factory=list, max_length=10)
    consensus_summary: str = Field(description="What all runs agree on.")
    manual_review_recommended: bool


_DEFAULT_CHAINS = {
    "narrate": (
        "You are a molecular biology analyst working with Phaseolus vulgaris "
        "salt-stress qPCR data.\n\n"
        "Target gene: {target_gene}\n\n"
        "Expression matrix (rows = salinity level, columns = time point, "
        "values = mean fold change RQ):\n{matrix}\n\n"
        "Precomputed facts (use these; do not recalculate):\n{stats}\n\n"
        "Interpretation rules: RQ > 1.0 indicates upregulation; RQ < 1.0 indicates "
        "transcriptional collapse. Report peak_time_point and peak_salinity exactly "
        "as given in the precomputed facts.\n\n"
        "Return ONLY valid JSON matching this schema:\n{schema}\n"
        "No markdown fences, no prose outside the JSON object."
    ),
    "adversarial": (
        "You are an adversarial scientific reviewer. Below are {n} independent "
        "summaries generated from the SAME {target_gene} qPCR dataset.\n\n"
        "Ground-truth facts computed directly from the data:\n{stats}\n\n"
        "Summaries:\n{summaries}\n\n"
        "Your job is to find contradictions BETWEEN the summaries, and any claim "
        "that conflicts with the ground-truth facts. Look specifically at: which "
        "condition is called the peak, whether expression is described as rising or "
        "falling, and any numeric values cited.\n\n"
        "Be skeptical. If the summaries genuinely agree, say so plainly.\n\n"
        "Return ONLY valid JSON matching this schema:\n{schema}\n"
        "No markdown fences, no prose outside the JSON object."
    ),
}


def load_prompt_chains(path: Path = PROMPT_CHAINS_PATH) -> dict:
    """Load prompt templates from prompt_chains.json, falling back to built-in defaults."""
    if path.is_file():
        with path.open("r", encoding="utf-8") as fh:
            chains = json.load(fh)
        missing = set(_DEFAULT_CHAINS) - set(chains)
        if missing:
            logger.warning("Prompt chains missing keys %s; filling from defaults", sorted(missing))
            chains = {**_DEFAULT_CHAINS, **chains}
        return chains
    logger.warning("%s not found; using built-in default prompts", path)
    return dict(_DEFAULT_CHAINS)


def _validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in SCHEMA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataframe is missing schema columns {missing}. "
            f"Expected the unified schema: {SCHEMA_COLUMNS}"
        )


def build_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to salinity x time_point, averaging technical replicates (spec Stage 2)."""
    matrix = df.pivot_table(
        index="salinity",
        columns="time_point",
        values="fold_change_rq",
        aggfunc="mean",
    )
    ordered = [s for s in SALINITY_LEVELS if s in matrix.index]
    extra = [s for s in matrix.index if s not in SALINITY_LEVELS]
    if extra:
        logger.warning("Unexpected salinity levels present: %s", extra)
    return matrix.reindex(ordered + extra)


def compute_facts(df: pd.DataFrame, target_gene: str) -> dict:
    """Precompute the numbers so the model narrates facts instead of doing arithmetic."""
    peak_row = df.loc[df["fold_change_rq"].idxmax()]
    trough_row = df.loc[df["fold_change_rq"].idxmin()]
    upregulated = int((df["fold_change_rq"] > 1.0).sum())
    collapsed = int((df["fold_change_rq"] < 1.0).sum())

    return {
        "target_gene": target_gene,
        "row_count": len(df),
        "time_points": sorted(df["time_point"].unique().tolist()),
        "salinity_levels": [s for s in SALINITY_LEVELS if s in set(df["salinity"])],
        "peak_fold_change": round(float(peak_row["fold_change_rq"]), 3),
        "peak_time_point": int(peak_row["time_point"]),
        "peak_salinity": str(peak_row["salinity"]),
        "min_fold_change": round(float(trough_row["fold_change_rq"]), 3),
        "min_time_point": int(trough_row["time_point"]),
        "min_salinity": str(trough_row["salinity"]),
        "mean_fold_change": round(float(df["fold_change_rq"].mean()), 3),
        "wells_upregulated": upregulated,
        "wells_collapsed": collapsed,
        "mean_variance_sd": round(float(df["variance_sd"].mean()), 3),
        "max_variance_sd": round(float(df["variance_sd"].max()), 3),
    }


def _format_facts(facts: dict) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in facts.items())


def narrate(
    df: pd.DataFrame,
    target_gene: str = "C5",
    provider: Optional[InferenceProvider] = None,
    chains: Optional[dict] = None,
    temperature: float = 0.2,
) -> NarrativeSummary:
    """Produce one schema-valid narrative summary, retrying once on invalid JSON."""
    _validate_schema(df)
    if df.empty:
        raise ValueError("Cannot narrate an empty dataframe")

    provider = provider or get_provider()
    chains = chains or load_prompt_chains()
    facts = compute_facts(df, target_gene)

    prompt = chains["narrate"].format(
        target_gene=target_gene,
        matrix=build_matrix(df).round(3).to_string(),
        stats=_format_facts(facts),
        schema=json.dumps(NarrativeSummary.model_json_schema(), indent=2),
    )

    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        content = provider.complete_json(prompt, temperature=temperature)
        try:
            return NarrativeSummary.model_validate_json(content)
        except ValidationError as exc:
            last_error = exc
            logger.warning("Narration attempt %d/%d failed validation: %s", attempt, MAX_ATTEMPTS, exc)
            logger.debug("Raw content: %s", content)

    raise ValueError(
        f"Model returned schema-invalid narration on all {MAX_ATTEMPTS} attempts"
    ) from last_error


def _pairwise_similarities(texts: list[str]) -> list[float]:
    return [
        difflib.SequenceMatcher(None, texts[i], texts[j]).ratio()
        for i in range(len(texts))
        for j in range(i + 1, len(texts))
    ]


def _claim_agreement(summaries: list[NarrativeSummary]) -> dict:
    """Fraction of runs agreeing on each factual claim, and whether it matches ground truth."""
    agreement = {}
    for key in ("peak_time_point", "peak_salinity", "direction"):
        values = [s.claims()[key] for s in summaries]
        modal = max(set(values), key=values.count)
        agreement[key] = {
            "modal_value": modal,
            "agreement_ratio": round(values.count(modal) / len(values), 4),
            "distinct_values": sorted({str(v) for v in values}),
        }
    return agreement


def adversarial_review(
    summaries: list[NarrativeSummary],
    facts: dict,
    target_gene: str,
    provider: InferenceProvider,
    chains: dict,
) -> Optional[ContradictionReport]:
    """Second-pass adversarial check for cross-run contradictions (spec Stage 3)."""
    rendered = "\n\n".join(
        f"--- Summary {i + 1} ---\n{json.dumps(s.model_dump(), indent=2)}"
        for i, s in enumerate(summaries)
    )
    prompt = chains["adversarial"].format(
        n=len(summaries),
        target_gene=target_gene,
        stats=_format_facts(facts),
        summaries=rendered,
        schema=json.dumps(ContradictionReport.model_json_schema(), indent=2),
    )

    for attempt in range(1, MAX_ATTEMPTS + 1):
        content = provider.complete_json(prompt, temperature=0.0)
        try:
            return ContradictionReport.model_validate_json(content)
        except ValidationError as exc:
            logger.warning("Adversarial attempt %d/%d failed validation: %s", attempt, MAX_ATTEMPTS, exc)
            logger.debug("Raw content: %s", content)

    logger.error("Adversarial review failed validation; falling back to variance scoring only")
    return None


def check_consistency(
    df: pd.DataFrame,
    target_gene: str = "C5",
    runs: int = CONSISTENCY_RUNS,
    provider: Optional[InferenceProvider] = None,
    temperature: float = 0.2,
) -> dict:
    """Narrate identical data `runs` times, then score lexical variance and contradictions.

    Returns a dashboard-ready report: confidence score, reliability rating, and a
    manual_review_required flag for Dr. Todd.
    """
    _validate_schema(df)
    provider = provider or get_provider()
    chains = load_prompt_chains()
    facts = compute_facts(df, target_gene)

    summaries: list[NarrativeSummary] = []
    failures = 0
    for run in range(1, runs + 1):
        logger.info("Consistency run %d/%d (%s)", run, runs, target_gene)
        try:
            summaries.append(narrate(df, target_gene, provider=provider, chains=chains, temperature=temperature))
        except (ValueError, RuntimeError) as exc:
            failures += 1
            logger.error("Run %d failed: %s", run, exc)

    if len(summaries) < 2:
        raise ValueError(f"Need at least 2 successful runs to measure variance; got {len(summaries)}")

    scores = _pairwise_similarities([s.as_text() for s in summaries])
    mean_sim = statistics.mean(scores)
    claim_agreement = _claim_agreement(summaries)

    # Ground-truth check: does the modal peak match what the data actually says?
    peak_correct = (
        claim_agreement["peak_time_point"]["modal_value"] == facts["peak_time_point"]
        and str(claim_agreement["peak_salinity"]["modal_value"]).lower() == facts["peak_salinity"].lower()
    )

    review = adversarial_review(summaries, facts, target_gene, provider, chains)

    # Confidence blends lexical stability, factual agreement, and the adversarial pass.
    claim_score = statistics.mean(c["agreement_ratio"] for c in claim_agreement.values())
    confidence = 0.4 * mean_sim + 0.4 * claim_score + 0.2 * (1.0 if peak_correct else 0.0)
    if review and review.contradictions_found:
        confidence *= 0.6

    if confidence >= 0.85:
        rating = "high"
    elif confidence >= 0.65:
        rating = "moderate"
    else:
        rating = "low"

    manual_review = (
        rating == "low"
        or not peak_correct
        or failures > 0
        or bool(review and review.manual_review_recommended)
    )

    report = {
        "target_gene": target_gene,
        "runs_requested": runs,
        "runs_succeeded": len(summaries),
        "runs_failed": failures,
        "lexical_variance": {
            "mean_similarity": round(mean_sim, 4),
            "stdev_similarity": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
            "min_similarity": round(min(scores), 4),
            "max_similarity": round(max(scores), 4),
        },
        "claim_agreement": claim_agreement,
        "peak_matches_ground_truth": peak_correct,
        "adversarial_review": review.model_dump() if review else None,
        "confidence_score": round(confidence, 4),
        "reliability_rating": rating,
        "manual_review_required": manual_review,
        "ground_truth_facts": facts,
        "summaries": [s.model_dump() for s in summaries],
    }

    logger.info(
        "%s: confidence=%.3f rating=%s peak_correct=%s contradictions=%s review=%s",
        target_gene, confidence, rating, peak_correct,
        review.contradictions_found if review else "n/a", manual_review,
    )
    return report


def run_consistency_report(
    input_file: Union[str, Path],
    output_file: Union[str, Path] = "consistency_report.json",
    target_gene: str = "C5",
    runs: int = CONSISTENCY_RUNS,
) -> dict:
    """Load Mike's cleaned CSV, run the verification loop, write a dashboard-ready report."""
    df = pd.read_csv(input_file)
    if "target_name" in df.columns:
        df = df[df["target_name"].astype(str).str.strip().str.upper() == target_gene.upper()]
        if df.empty:
            raise ValueError(f"No rows for target gene '{target_gene}' in {input_file}")

    report = check_consistency(df, target_gene=target_gene, runs=runs)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Wrote consistency report to %s", output_path)
    return report


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="ASTRA narrator: narrate and verify qPCR expression data.")
    ap.add_argument("input_file", nargs="?", default="parsed_output.csv")
    ap.add_argument("-g", "--gene", default="C5", help="Target gene (C5, C3, Actin, ...)")
    ap.add_argument("-o", "--output", default="consistency_report.json")
    ap.add_argument("-n", "--runs", type=int, default=CONSISTENCY_RUNS)
    args = ap.parse_args()

    result = run_consistency_report(args.input_file, args.output, args.gene, args.runs)
    print(f"\n{args.gene}: {result['reliability_rating']} confidence "
          f"({result['confidence_score']:.3f}) | manual review: {result['manual_review_required']}")