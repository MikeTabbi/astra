"""Generate validated narrative summaries from ASTRA parser output.

The narrator consumes the exact five-column CSV produced by ``parser.py``:

    target_name, time_point, salinity, fold_change_rq, variance_sd

It validates that boundary, computes deterministic facts, asks an inference
provider for a Pydantic-structured summary, verifies the model's peak claim
against the source data, and repeats the generation five times to measure
narrative consistency.
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import math
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence, Union

import pandas as pd
from ollama import Client
from pydantic import BaseModel, Field, ValidationError


PathLike = Union[str, Path]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")
DEFAULT_INPUT_FILE = Path("parsed_output.csv")
DEFAULT_OUTPUT_FILE = Path("consistency_report.json")
PROMPT_CHAINS_PATH = Path(__file__).with_name("prompt_chains.json")

MAX_ATTEMPTS = 2
CONSISTENCY_RUNS = 5
PEAK_RQ_ABS_TOLERANCE = 0.001
UNSUPPORTED_CLAIM_PATTERNS = {
    r"\bsignificant(?:ly)?\b": "hypothesis-test language is unsupported",
    r"\bstatistically\b": "hypothesis-test language is unsupported",
    r"\bcaus(?:e|es|ed|ing)\b": "causality is not established",
    r"\bleads? to\b": "causality is not established",
    r"\bin response to\b": "causality is not established",
    r"\bsingle replicate\b": (
        "the parser output does not establish the biological replicate count"
    ),
}

PARSER_SCHEMA_COLUMNS = [
    "target_name",
    "time_point",
    "salinity",
    "fold_change_rq",
    "variance_sd",
]
NUMERIC_COLUMNS = [
    "time_point",
    "fold_change_rq",
    "variance_sd",
]


class DatasetFacts(BaseModel):
    """Facts calculated directly from the parser output, without an LLM."""

    target_name: str
    row_count: int = Field(ge=1)
    time_points: list[float] = Field(min_length=1)
    salinity_conditions: list[str] = Field(min_length=1)
    fold_change_min: float
    fold_change_max: float
    fold_change_mean: float
    variance_sd_mean: float
    peak_time_point: float
    peak_salinity: str
    peak_fold_change_rq: float = Field(ge=0)


class NarrativeSummary(BaseModel):
    """Pydantic-enforced response contract for one model narration."""

    target_name: str = Field(description="Target gene summarized by this response.")
    headline: str = Field(description="One-sentence takeaway grounded in the data.")
    trend_description: str = Field(
        description="Observed expression pattern across time points and conditions."
    )
    peak_condition: str = Field(
        description="Plain-language description of the strongest measured response."
    )
    peak_time_point: float = Field(
        ge=0,
        description="Time point of the maximum fold_change_rq.",
    )
    peak_salinity: str = Field(
        description="Categorical salinity/experimental condition at the peak."
    )
    peak_fold_change_rq: float = Field(
        ge=0,
        description="Maximum fold_change_rq present in the source table.",
    )
    confidence_note: str = Field(
        description=(
            "A cautious limitation note based on variance and dataset coverage; "
            "not a statistical confidence claim."
        )
    )
    key_observations: list[str] = Field(min_length=1, max_length=5)

    def as_text(self) -> str:
        """Flatten narrative fields for wording-consistency comparison."""
        return " ".join(
            [
                self.headline,
                self.trend_description,
                self.peak_condition,
                self.confidence_note,
                *self.key_observations,
            ]
        )

    def peak_key(self) -> tuple[str, float, str, float]:
        """Return a normalized peak identity for cross-run comparisons."""
        return (
            self.target_name.strip().casefold(),
            float(self.peak_time_point),
            self.peak_salinity.strip().casefold(),
            round(float(self.peak_fold_change_rq), 6),
        )


class InferenceProvider(Protocol):
    """Backend-neutral interface for local Ollama or a future cloud provider."""

    def generate_narration(
        self,
        *,
        prompt: str,
        schema: dict[str, Any],
        model: str,
        temperature: float,
    ) -> str:
        """Return one JSON string matching ``schema``."""


class OllamaInferenceProvider:
    """Inference provider backed by an Ollama server."""

    def __init__(
        self,
        host: str = DEFAULT_OLLAMA_HOST,
        *,
        client: Optional[Client] = None,
        verify_connection: bool = True,
    ) -> None:
        self.host = host
        self.client = client or Client(host=host)
        if verify_connection:
            try:
                self.client.list()
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot reach Ollama at {host}. Confirm the host is running "
                    "'OLLAMA_HOST=0.0.0.0 ollama serve', both machines are on the "
                    "same LAN, and port 11434 is open."
                ) from exc
            logger.info("Connected to Ollama at %s", host)

    def generate_narration(
        self,
        *,
        prompt: str,
        schema: dict[str, Any],
        model: str,
        temperature: float,
    ) -> str:
        response = self.client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format=schema,
            options={"temperature": temperature},
        )
        return _extract_ollama_content(response)


def _extract_ollama_content(response: object) -> str:
    """Read content from either dict-like or object-like Ollama responses."""
    try:
        message = response["message"]  # type: ignore[index]
    except (KeyError, TypeError):
        message = getattr(response, "message", None)

    if message is None:
        raise RuntimeError(f"Unexpected response shape from Ollama: {response!r}")

    try:
        content = message["content"]
    except (KeyError, TypeError):
        content = getattr(message, "content", None)

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"Ollama returned no narrative content: {response!r}")
    return content


def load_prompt_chains(path: Path = PROMPT_CHAINS_PATH) -> dict[str, str]:
    """Load a custom prompt or return the parser-aligned built-in prompt."""
    if path.is_file():
        with path.open("r", encoding="utf-8") as file_handle:
            chains = json.load(file_handle)
        if not isinstance(chains, dict) or not isinstance(chains.get("narrate"), str):
            raise ValueError(
                f"{path} must contain a string value under the 'narrate' key"
            )
        return chains

    logger.warning("%s not found; using built-in default prompt", path)
    return {
        "narrate": (
            "You are a cautious molecular-biology data narrator. Summarize only "
            "the measurements supplied for {target_gene}. Use only descriptive "
            "comparisons supported by the supplied numbers. Do not claim "
            "hypothesis-test results, causality, future behavior, or unmeasured "
            "biology.\n\n"
            "ASTRA parser data contract:\n"
            "- time_point is the experimental time value supplied by the researcher.\n"
            "- salinity is a categorical condition label such as Low, High, or "
            "Control. It is NOT a numeric ppt measurement.\n"
            "- fold_change_rq is relative expression.\n"
            "- variance_sd is technical-replicate standard deviation.\n\n"
            "Each row is a cleaned sample-level result after parser QC and "
            "technical-replicate aggregation. Say 'higher measured fold change' "
            "or 'lower measured fold change' when comparing values. Do not infer "
            "a biological-replicate count.\n\n"
            "Clean parser output:\n{data_table}\n\n"
            "Deterministic facts calculated in Python:\n{stats}\n\n"
            "Your structured peak fields must exactly match the deterministic peak. "
            "Describe uncertainty cautiously; consistency is not scientific "
            "confidence. Return ONLY valid JSON matching this schema:\n{schema}\n"
            "Do not use markdown fences or add prose outside the JSON."
        )
    }


def _invalid_row_numbers(mask: pd.Series) -> list[int]:
    """Convert a validation mask to human-readable CSV row numbers."""
    return [int(index) + 2 for index in mask[mask].index]


def validate_parser_dataframe(
    dataframe: pd.DataFrame,
    *,
    target_gene: str = "C5",
) -> pd.DataFrame:
    """Validate and isolate one target from parser.py's five-column contract."""
    missing = [
        column for column in PARSER_SCHEMA_COLUMNS if column not in dataframe.columns
    ]
    if missing:
        raise ValueError(
            "Narrator input is not valid parser output. Missing column(s): "
            + ", ".join(missing)
        )

    data = dataframe[PARSER_SCHEMA_COLUMNS].copy()
    if data.empty:
        raise ValueError("Narrator input contains no accepted parser rows")

    for column in ("target_name", "salinity"):
        data[column] = data[column].astype("string").str.strip()
        invalid = data[column].isna() | data[column].eq("")
        if invalid.any():
            raise ValueError(
                f"Column '{column}' is blank at CSV row(s): "
                f"{_invalid_row_numbers(invalid)}"
            )

    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        invalid = data[column].isna() | ~data[column].map(
            lambda value: math.isfinite(float(value)) if pd.notna(value) else False
        )
        if invalid.any():
            raise ValueError(
                f"Column '{column}' is not numeric at CSV row(s): "
                f"{_invalid_row_numbers(invalid)}"
            )

    negative_time = data["time_point"] < 0
    negative_fold_change = data["fold_change_rq"] < 0
    negative_variance = data["variance_sd"] < 0
    for label, invalid in (
        ("time_point", negative_time),
        ("fold_change_rq", negative_fold_change),
        ("variance_sd", negative_variance),
    ):
        if invalid.any():
            raise ValueError(
                f"Column '{label}' cannot be negative at CSV row(s): "
                f"{_invalid_row_numbers(invalid)}"
            )

    requested_target = target_gene.strip()
    target_mask = data["target_name"].str.casefold().eq(requested_target.casefold())
    data = data.loc[target_mask].copy()
    if data.empty:
        available = sorted(
            {
                str(value)
                for value in dataframe["target_name"].dropna().astype(str)
                if value.strip()
            },
            key=str.casefold,
        )
        raise ValueError(
            f"No rows found for target '{requested_target}'. "
            f"Available targets: {', '.join(available) or '<none>'}"
        )

    data["target_name"] = requested_target.upper()
    return data.reset_index(drop=True)


def load_parser_output(
    input_file: PathLike = DEFAULT_INPUT_FILE,
    *,
    target_gene: str = "C5",
) -> pd.DataFrame:
    """Read and validate the clean CSV produced by parser.py."""
    path = Path(input_file)
    if not path.is_file():
        raise FileNotFoundError(
            f"Clean parser output not found: {path}. Run parser.py first."
        )
    return validate_parser_dataframe(pd.read_csv(path), target_gene=target_gene)


def compute_dataset_facts(
    dataframe: pd.DataFrame,
    *,
    target_gene: str = "C5",
) -> DatasetFacts:
    """Calculate the factual values the model is required to preserve."""
    data = validate_parser_dataframe(dataframe, target_gene=target_gene)
    peak = data.loc[data["fold_change_rq"].idxmax()]
    time_points = sorted({float(value) for value in data["time_point"]})
    salinity_conditions = sorted(
        {str(value) for value in data["salinity"]},
        key=str.casefold,
    )
    return DatasetFacts(
        target_name=target_gene.strip().upper(),
        row_count=len(data),
        time_points=time_points,
        salinity_conditions=salinity_conditions,
        fold_change_min=float(data["fold_change_rq"].min()),
        fold_change_max=float(data["fold_change_rq"].max()),
        fold_change_mean=float(data["fold_change_rq"].mean()),
        variance_sd_mean=float(data["variance_sd"].mean()),
        peak_time_point=float(peak["time_point"]),
        peak_salinity=str(peak["salinity"]),
        peak_fold_change_rq=float(peak["fold_change_rq"]),
    )


def _display_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _summarize_dataframe(
    dataframe: pd.DataFrame,
    *,
    target_gene: str = "C5",
) -> str:
    """Create a compact, deterministic fact block for the model prompt."""
    facts = compute_dataset_facts(dataframe, target_gene=target_gene)
    time_values = ", ".join(_display_number(value) for value in facts.time_points)
    conditions = ", ".join(facts.salinity_conditions)
    lines = [
        f"- target_name: {facts.target_name}\n"
        f"- accepted rows: {facts.row_count}\n"
        f"- measured time_point value(s): {time_values}\n"
        f"- categorical salinity/condition labels: {conditions}\n"
        f"- fold_change_rq range: {facts.fold_change_min:.3f} to "
        f"{facts.fold_change_max:.3f}\n"
        f"- mean fold_change_rq: {facts.fold_change_mean:.3f}\n"
        f"- measured peak: {facts.peak_fold_change_rq:.3f} at "
        f"time_point={_display_number(facts.peak_time_point)}, "
        f"salinity={facts.peak_salinity}\n"
        f"- mean variance_sd: {facts.variance_sd_mean:.3f}\n"
    ]
    if len(facts.time_points) == 1:
        lines.append(
            "- temporal coverage: only one measured time point; no time-course "
            "trend can be inferred\n"
        )

    data = validate_parser_dataframe(dataframe, target_gene=target_gene)
    for condition, group in data.groupby("salinity", sort=True):
        lines.append(
            f"- condition {condition}: rows={len(group)}, "
            f"mean fold_change_rq={group['fold_change_rq'].mean():.3f}, "
            f"range={group['fold_change_rq'].min():.3f} to "
            f"{group['fold_change_rq'].max():.3f}\n"
        )
    return "".join(lines)


def build_prompt(
    dataframe: pd.DataFrame,
    target_gene: str = "C5",
    chains: Optional[dict[str, str]] = None,
) -> str:
    """Build a prompt from validated parser rows and deterministic facts."""
    data = validate_parser_dataframe(dataframe, target_gene=target_gene)
    chains = chains or load_prompt_chains()
    return chains["narrate"].format(
        target_gene=target_gene.strip().upper(),
        data_table=data.to_csv(index=False),
        stats=_summarize_dataframe(data, target_gene=target_gene),
        schema=json.dumps(NarrativeSummary.model_json_schema(), indent=2),
    )


def _validate_summary_facts(
    summary: NarrativeSummary,
    facts: DatasetFacts,
) -> None:
    """Reject a structured model response that contradicts measured peak facts."""
    errors: list[str] = []
    if summary.target_name.strip().casefold() != facts.target_name.casefold():
        errors.append(
            f"target_name={summary.target_name!r}, expected {facts.target_name!r}"
        )
    if not math.isclose(
        float(summary.peak_time_point),
        facts.peak_time_point,
        rel_tol=0,
        abs_tol=1e-9,
    ):
        errors.append(
            f"peak_time_point={summary.peak_time_point}, "
            f"expected {facts.peak_time_point}"
        )
    if summary.peak_salinity.strip().casefold() != facts.peak_salinity.casefold():
        errors.append(
            f"peak_salinity={summary.peak_salinity!r}, "
            f"expected {facts.peak_salinity!r}"
        )
    if not math.isclose(
        float(summary.peak_fold_change_rq),
        facts.peak_fold_change_rq,
        rel_tol=0,
        abs_tol=PEAK_RQ_ABS_TOLERANCE,
    ):
        errors.append(
            f"peak_fold_change_rq={summary.peak_fold_change_rq}, "
            f"expected {facts.peak_fold_change_rq}"
        )
    if " ppt" in summary.as_text().casefold():
        errors.append("narrative treats categorical salinity labels as numeric ppt")

    if errors:
        raise ValueError("Narrative factual validation failed: " + "; ".join(errors))


def _normalize_deterministic_fields(
    summary: NarrativeSummary,
    facts: DatasetFacts,
) -> NarrativeSummary:
    """Replace factual prose fields with values calculated directly in Python."""
    summary.peak_condition = (
        f"The strongest measured C5 response was "
        f"{facts.peak_fold_change_rq:.3f} in the {facts.peak_salinity} condition "
        f"at time point {_display_number(facts.peak_time_point)}."
    )
    coverage = (
        "Only one time point is represented, so this report cannot describe a "
        "time-course trend. "
        if len(facts.time_points) == 1
        else ""
    )
    summary.confidence_note = (
        f"{coverage}The report is descriptive: variance_sd reflects technical-"
        "replicate spread, and no statistical significance or causality was tested."
    )
    return summary


def _validate_narrative_language(summary: NarrativeSummary) -> None:
    """Reject confident scientific claims that the parser output cannot support."""
    generated_text = " ".join(
        [
            summary.headline,
            summary.trend_description,
            *summary.key_observations,
        ]
    ).casefold()
    violations = [
        explanation
        for pattern, explanation in UNSUPPORTED_CLAIM_PATTERNS.items()
        if re.search(pattern, generated_text)
    ]
    if violations:
        raise ValueError(
            "Unsupported narrative claim(s): " + "; ".join(sorted(set(violations)))
        )


def narrate(
    dataframe: pd.DataFrame,
    target_gene: str = "C5",
    *,
    provider: Optional[InferenceProvider] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
) -> NarrativeSummary:
    """Generate one schema-valid, peak-verified narrative summary."""
    data = validate_parser_dataframe(dataframe, target_gene=target_gene)
    facts = compute_dataset_facts(data, target_gene=target_gene)
    provider = provider or OllamaInferenceProvider()
    prompt = build_prompt(data, target_gene)
    schema = NarrativeSummary.model_json_schema()
    last_error: Optional[Exception] = None
    attempt_prompt = prompt

    for attempt in range(1, MAX_ATTEMPTS + 1):
        content = provider.generate_narration(
            prompt=attempt_prompt,
            schema=schema,
            model=model,
            temperature=temperature,
        )
        try:
            summary = NarrativeSummary.model_validate_json(content)
            _validate_summary_facts(summary, facts)
            _validate_narrative_language(summary)
            return _normalize_deterministic_fields(summary, facts)
        except (ValidationError, ValueError) as exc:
            last_error = exc
            logger.warning(
                "Narration attempt %d/%d failed validation: %s",
                attempt,
                MAX_ATTEMPTS,
                exc,
            )
            attempt_prompt = (
                f"{prompt}\n\nCORRECTION REQUIRED: The previous response was "
                f"rejected because {exc}. Return a corrected JSON response that "
                "uses only descriptive, non-causal language."
            )

    raise ValueError(
        f"Model returned invalid or factually inconsistent output on all "
        f"{MAX_ATTEMPTS} attempts"
    ) from last_error


def _pairwise_similarities(texts: list[str]) -> list[float]:
    """Calculate a wording-similarity ratio for every unordered pair."""
    scores: list[float] = []
    for first in range(len(texts)):
        for second in range(first + 1, len(texts)):
            scores.append(
                difflib.SequenceMatcher(
                    None,
                    texts[first],
                    texts[second],
                ).ratio()
            )
    return scores


def check_consistency(
    dataframe: pd.DataFrame,
    target_gene: str = "C5",
    runs: int = CONSISTENCY_RUNS,
    *,
    provider: Optional[InferenceProvider] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
) -> dict[str, Any]:
    """Run repeated narrations and report consistency, not scientific confidence."""
    if runs < 2:
        raise ValueError("Consistency checking requires at least 2 runs")

    data = validate_parser_dataframe(dataframe, target_gene=target_gene)
    facts = compute_dataset_facts(data, target_gene=target_gene)
    provider = provider or OllamaInferenceProvider()
    summaries: list[NarrativeSummary] = []
    failures = 0

    for run in range(1, runs + 1):
        logger.info("Consistency run %d/%d", run, runs)
        try:
            summaries.append(
                narrate(
                    data,
                    target_gene,
                    provider=provider,
                    model=model,
                    temperature=temperature,
                )
            )
        except (ValueError, RuntimeError) as exc:
            failures += 1
            logger.error("Consistency run %d failed: %s", run, exc)

    if len(summaries) < 2:
        raise ValueError(
            "Need at least 2 successful runs to measure consistency; "
            f"got {len(summaries)}"
        )

    texts = [summary.as_text() for summary in summaries]
    scores = _pairwise_similarities(texts)
    mean_similarity = statistics.mean(scores)
    stdev_similarity = statistics.stdev(scores) if len(scores) > 1 else 0.0

    peak_keys = [summary.peak_key() for summary in summaries]
    most_common_peak = max(set(peak_keys), key=peak_keys.count)
    peak_agreement = peak_keys.count(most_common_peak) / len(peak_keys)

    if peak_agreement < 1.0 or mean_similarity < 0.65:
        rating = "low"
    elif mean_similarity >= 0.85:
        rating = "high"
    else:
        rating = "moderate"

    report: dict[str, Any] = {
        "input_schema": PARSER_SCHEMA_COLUMNS,
        "dataset_facts": facts.model_dump(),
        "runs_requested": runs,
        "runs_succeeded": len(summaries),
        "runs_failed": failures,
        "mean_similarity": round(mean_similarity, 4),
        "stdev_similarity": round(stdev_similarity, 4),
        "min_similarity": round(min(scores), 4),
        "max_similarity": round(max(scores), 4),
        "reliability_rating": rating,
        "reliability_scope": (
            "Generation consistency only; this is not statistical confidence "
            "or biological validation."
        ),
        "peak_condition_agreement": round(peak_agreement, 4),
        "summaries": [summary.model_dump() for summary in summaries],
    }
    logger.info(
        "Consistency: mean=%.3f stdev=%.3f peak_agreement=%.3f rating=%s",
        mean_similarity,
        stdev_similarity,
        peak_agreement,
        rating,
    )
    return report


def run_consistency_report(
    input_file: PathLike = DEFAULT_INPUT_FILE,
    output_file: PathLike = DEFAULT_OUTPUT_FILE,
    target_gene: str = "C5",
    runs: int = CONSISTENCY_RUNS,
    *,
    provider: Optional[InferenceProvider] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
) -> dict[str, Any]:
    """Load parser output, run consistency checks, and write a JSON report."""
    dataframe = load_parser_output(input_file, target_gene=target_gene)
    report = check_consistency(
        dataframe,
        target_gene=target_gene,
        runs=runs,
        provider=provider,
        model=model,
        temperature=temperature,
    )

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(report, file_handle, indent=2)
    logger.info("Wrote consistency report to %s", output_path)
    return report


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Narrate the clean five-column CSV produced by parser.py."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=str(DEFAULT_INPUT_FILE),
        help=f"Parser output CSV (default: {DEFAULT_INPUT_FILE})",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(DEFAULT_OUTPUT_FILE),
        help=f"JSON report path (default: {DEFAULT_OUTPUT_FILE})",
    )
    parser.add_argument(
        "-t",
        "--target",
        default="C5",
        help="Target gene to narrate (default: C5)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=CONSISTENCY_RUNS,
        help=f"Number of consistency runs, minimum 2 (default: {CONSISTENCY_RUNS})",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_OLLAMA_HOST,
        help=f"Ollama server URL (default: {DEFAULT_OLLAMA_HOST})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Model generation temperature (default: 0.2)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point."""
    arguments = _build_cli_parser().parse_args(argv)
    try:
        provider = OllamaInferenceProvider(host=arguments.host)
        report = run_consistency_report(
            input_file=arguments.input_file,
            output_file=arguments.output,
            target_gene=arguments.target,
            runs=arguments.runs,
            provider=provider,
            model=arguments.model,
            temperature=arguments.temperature,
        )
    except Exception as exc:
        logger.error("ASTRA narrator failed: %s", exc)
        return 1

    print("\nASTRA narrator complete")
    print(f"Successful runs: {report['runs_succeeded']}/{report['runs_requested']}")
    print(f"Consistency rating: {report['reliability_rating'].upper()}")
    print(f"Mean wording similarity: {report['mean_similarity']}")
    print(f"Peak agreement: {report['peak_condition_agreement']}")
    print(f"Report: {arguments.output}")
    print("\nValidated sample summary:")
    print(json.dumps(report["summaries"][0], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
