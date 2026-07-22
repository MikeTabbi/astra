"""Narrative summarization of qPCR expression data via a local Ollama model.

Objectives:
  1. Pydantic-enforced JSON output from the LLM.
  2. LAN connection to the team's host machine running Ollama.
  3. 5x consecutive consistency loop measuring text variance across runs.
"""

import difflib
import json
import logging
import os
import statistics
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from ollama import Client
from pydantic import BaseModel, Field, ValidationError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://192.168.1.100:11434")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")

MAX_ATTEMPTS = 2
CONSISTENCY_RUNS = 5

PROMPT_CHAINS_PATH = Path("prompt_chains.json")


class NarrativeSummary(BaseModel):
    """Structured narration of a qPCR expression dataset."""

    headline: str = Field(description="One-sentence takeaway.")
    trend_description: str = Field(description="How expression changes across conditions.")
    peak_condition: str = Field(description="Time point and salinity where response is strongest.")
    confidence_note: str = Field(description="Caveats given the variance in the data.")
    key_observations: list[str] = Field(min_length=1, max_length=5)

    def as_text(self) -> str:
        """Flatten to a single string for variance comparison."""
        return " ".join([
            self.headline,
            self.trend_description,
            self.peak_condition,
            self.confidence_note,
            *self.key_observations,
        ])


def load_prompt_chains(path: Path = PROMPT_CHAINS_PATH) -> dict:
    """Load prompt templates from prompt_chains.json, falling back to a built-in default."""
    if path.is_file():
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    logger.warning("%s not found; using built-in default prompt", path)
    return {
        "narrate": (
            "You are a molecular biology analyst. Summarize the following qPCR "
            "results for the {target_gene} gene.\n\n"
            "Data (time_point in hours, salinity in ppt, fold_change_rq is relative "
            "expression, variance_sd is standard deviation):\n{data_table}\n\n"
            "Summary statistics:\n{stats}\n\n"
            "Return ONLY valid JSON matching this schema:\n{schema}\n"
            "No markdown fences, no prose outside the JSON."
        )
    }


def _summarize_dataframe(df: pd.DataFrame) -> str:
    """Compact numeric summary so the model reasons over facts, not just raw rows."""
    peak = df.loc[df["fold_change_rq"].idxmax()]
    return (
        f"- rows: {len(df)}\n"
        f"- time_point range: {df['time_point'].min()} to {df['time_point'].max()} hrs\n"
        f"- salinity range: {df['salinity'].min()} to {df['salinity'].max()} ppt\n"
        f"- fold_change_rq range: {df['fold_change_rq'].min():.3f} to {df['fold_change_rq'].max():.3f}\n"
        f"- mean fold_change_rq: {df['fold_change_rq'].mean():.3f}\n"
        f"- peak response: {peak['fold_change_rq']:.3f} at "
        f"time_point={peak['time_point']}, salinity={peak['salinity']}\n"
        f"- mean variance_sd: {df['variance_sd'].mean():.3f}\n"
    )


def build_prompt(df: pd.DataFrame, target_gene: str = "C5", chains: Optional[dict] = None) -> str:
    if chains is None:
        chains = load_prompt_chains()
    return chains["narrate"].format(
        target_gene=target_gene,
        data_table=df.to_csv(index=False),
        stats=_summarize_dataframe(df),
        schema=json.dumps(NarrativeSummary.model_json_schema(), indent=2),
    )


def get_client(host: str = DEFAULT_OLLAMA_HOST) -> Client:
    """Create an Ollama client pointed at the team host, verifying reachability."""
    client = Client(host=host)
    try:
        client.list()
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {host}. Check the host machine is running "
            f"'OLLAMA_HOST=0.0.0.0 ollama serve', that both machines are on the "
            f"same LAN, and that port 11434 is open."
        ) from exc
    logger.info("Connected to Ollama at %s", host)
    return client


def narrate(
    df: pd.DataFrame,
    target_gene: str = "C5",
    client: Optional[Client] = None,
    model: str = MODEL_NAME,
    temperature: float = 0.2,
) -> NarrativeSummary:
    """Produce one structured narrative summary of `df`, retrying on invalid JSON."""
    if df.empty:
        raise ValueError("Cannot narrate an empty dataframe")

    client = client or get_client()
    prompt = build_prompt(df, target_gene)
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": temperature},
        )
        try:
            content = response["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Unexpected response shape from Ollama: {response!r}") from exc

        try:
            return NarrativeSummary.model_validate_json(content)
        except ValidationError as exc:
            last_error = exc
            logger.warning("Attempt %d/%d failed schema validation: %s", attempt, MAX_ATTEMPTS, exc)
            logger.debug("Raw content was: %s", content)

    raise ValueError(f"Model returned schema-invalid output on all {MAX_ATTEMPTS} attempts") from last_error


def _pairwise_similarities(texts: list[str]) -> list[float]:
    """Similarity ratio for every unordered pair of narration texts."""
    scores = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            scores.append(difflib.SequenceMatcher(None, texts[i], texts[j]).ratio())
    return scores


def check_consistency(
    df: pd.DataFrame,
    target_gene: str = "C5",
    runs: int = CONSISTENCY_RUNS,
    client: Optional[Client] = None,
    model: str = MODEL_NAME,
    temperature: float = 0.2,
) -> dict:
    """Narrate the same data `runs` times and score how much the wording varies.

    Returns a dict with per-run summaries, pairwise similarity stats, and a
    reliability rating derived from mean similarity.
    """
    client = client or get_client()
    summaries: list[NarrativeSummary] = []
    failures = 0

    for run in range(1, runs + 1):
        logger.info("Consistency run %d/%d", run, runs)
        try:
            summaries.append(narrate(df, target_gene, client=client, model=model, temperature=temperature))
        except (ValueError, RuntimeError) as exc:
            failures += 1
            logger.error("Run %d failed: %s", run, exc)

    if len(summaries) < 2:
        raise ValueError(f"Need at least 2 successful runs to measure variance; got {len(summaries)}")

    texts = [s.as_text() for s in summaries]
    scores = _pairwise_similarities(texts)
    mean_sim = statistics.mean(scores)
    stdev_sim = statistics.stdev(scores) if len(scores) > 1 else 0.0

    if mean_sim >= 0.85:
        rating = "high"
    elif mean_sim >= 0.65:
        rating = "moderate"
    else:
        rating = "low"

    peaks = [s.peak_condition for s in summaries]
    peak_agreement = peaks.count(max(set(peaks), key=peaks.count)) / len(peaks)

    report = {
        "runs_requested": runs,
        "runs_succeeded": len(summaries),
        "runs_failed": failures,
        "mean_similarity": round(mean_sim, 4),
        "stdev_similarity": round(stdev_sim, 4),
        "min_similarity": round(min(scores), 4),
        "max_similarity": round(max(scores), 4),
        "reliability_rating": rating,
        "peak_condition_agreement": round(peak_agreement, 4),
        "summaries": [s.model_dump() for s in summaries],
    }

    logger.info(
        "Consistency: mean=%.3f stdev=%.3f rating=%s (%d/%d runs succeeded)",
        mean_sim, stdev_sim, rating, len(summaries), runs,
    )
    return report


def run_consistency_report(
    input_file: Union[str, Path],
    output_file: Union[str, Path] = "consistency_report.json",
    target_gene: str = "C5",
    runs: int = CONSISTENCY_RUNS,
) -> dict:
    """Load cleaned qPCR data, run the consistency loop, and write the report to JSON."""
    df = pd.read_csv(input_file)
    report = check_consistency(df, target_gene=target_gene, runs=runs)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Wrote consistency report to %s", output_path)
    return report


if __name__ == "__main__":
    run_consistency_report("parsed_output.csv")