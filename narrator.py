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
from pydantic import BaseModel, Field, PrivateAttr, ValidationError


PathLike = Union[str, Path]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
DEFAULT_INPUT_FILE = Path("parsed_output.csv")
DEFAULT_OUTPUT_FILE = Path("consistency_report.json")
DEFAULT_PARSED_DIRECTORY = PROJECT_ROOT / "datasets" / "parsed"
DEFAULT_REPORT_DIRECTORY = PROJECT_ROOT / "reports"
DEFAULT_TARGET_GENE = "auto"
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
    condition_means: dict[str, float] = Field(min_length=1)
    fold_change_min: float
    fold_change_max: float
    fold_change_mean: float
    variance_sd_mean: float
    peak_time_point: float
    peak_salinity: str
    peak_fold_change_rq: float = Field(ge=0)
    highest_mean_salinity: str
    highest_mean_fold_change_rq: float = Field(ge=0)


class NarrativeSummary(BaseModel):
    """Pydantic-enforced response contract for one model narration."""

    _generated_text: str = PrivateAttr(default="")

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
    highest_mean_salinity: str = Field(
        description="Condition with the highest mean fold_change_rq."
    )
    highest_mean_fold_change_rq: float = Field(
        ge=0,
        description="Mean fold_change_rq for highest_mean_salinity.",
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

    def consistency_text(self) -> str:
        """Return the original validated model wording for consistency scoring."""
        return self._generated_text or self.as_text()


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
            "technical-replicate aggregation. Keep the strongest individual "
            "measurement separate from the condition with the highest mean. "
            "When discussing conditions, report their calculated means instead "
            "of writing free-form higher/lower comparisons. Every superlative "
            "must explicitly say either 'condition mean' or 'individual "
            "measurement'. Do not infer a biological-replicate count.\n\n"
            "Clean parser output:\n{data_table}\n\n"
            "Deterministic facts calculated in Python:\n{stats}\n\n"
            "Your structured peak fields and highest-condition-mean fields must "
            "exactly match the deterministic facts. Describe uncertainty "
            "cautiously; consistency is not scientific confidence. Return ONLY "
            "valid JSON matching this schema:\n{schema}\n"
            "Do not use markdown fences or add prose outside the JSON."
        )
    }


def _invalid_row_numbers(mask: pd.Series) -> list[int]:
    """Convert a validation mask to human-readable CSV row numbers."""
    return [int(index) + 2 for index in mask[mask].index]


def _resolve_target_name(dataframe: pd.DataFrame, target_gene: str) -> str:
    """Resolve ``auto`` to the single target present in parser output."""
    available = sorted(
        {
            str(value).strip()
            for value in dataframe["target_name"].dropna().astype(str)
            if value.strip()
        },
        key=str.casefold,
    )
    requested = target_gene.strip()
    if requested.casefold() != DEFAULT_TARGET_GENE.casefold():
        return requested
    if len(available) != 1:
        raise ValueError(
            "Automatic target detection requires exactly one target; "
            f"available targets: {', '.join(available) or '<none>'}"
        )
    return available[0]


def validate_parser_dataframe(
    dataframe: pd.DataFrame,
    *,
    target_gene: str = DEFAULT_TARGET_GENE,
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

    requested_target = _resolve_target_name(data, target_gene)
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

    data["target_name"] = requested_target
    return data.reset_index(drop=True)


def load_parser_output(
    input_file: PathLike = DEFAULT_INPUT_FILE,
    *,
    target_gene: str = DEFAULT_TARGET_GENE,
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
    target_gene: str = DEFAULT_TARGET_GENE,
) -> DatasetFacts:
    """Calculate the factual values the model is required to preserve."""
    data = validate_parser_dataframe(dataframe, target_gene=target_gene)
    peak = data.loc[data["fold_change_rq"].idxmax()]
    time_points = sorted({float(value) for value in data["time_point"]})
    salinity_conditions = sorted(
        {str(value) for value in data["salinity"]},
        key=str.casefold,
    )
    condition_means = {
        str(condition): float(mean)
        for condition, mean in data.groupby("salinity", sort=True)[
            "fold_change_rq"
        ].mean().items()
    }
    highest_mean_salinity = max(
        condition_means,
        key=lambda condition: condition_means[condition],
    )
    resolved_target = str(data.iloc[0]["target_name"])
    return DatasetFacts(
        target_name=resolved_target,
        row_count=len(data),
        time_points=time_points,
        salinity_conditions=salinity_conditions,
        condition_means=condition_means,
        fold_change_min=float(data["fold_change_rq"].min()),
        fold_change_max=float(data["fold_change_rq"].max()),
        fold_change_mean=float(data["fold_change_rq"].mean()),
        variance_sd_mean=float(data["variance_sd"].mean()),
        peak_time_point=float(peak["time_point"]),
        peak_salinity=str(peak["salinity"]),
        peak_fold_change_rq=float(peak["fold_change_rq"]),
        highest_mean_salinity=highest_mean_salinity,
        highest_mean_fold_change_rq=condition_means[highest_mean_salinity],
    )


def _display_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _condition_means_text(facts: DatasetFacts) -> str:
    """Render condition means in a stable, auditable order."""
    return ", ".join(
        f"{condition}={mean:.3f}"
        for condition, mean in facts.condition_means.items()
    )


def _summarize_dataframe(
    dataframe: pd.DataFrame,
    *,
    target_gene: str = DEFAULT_TARGET_GENE,
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
        f"- highest condition mean: {facts.highest_mean_fold_change_rq:.3f} "
        f"for salinity={facts.highest_mean_salinity}\n"
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
    target_gene: str = DEFAULT_TARGET_GENE,
    chains: Optional[dict[str, str]] = None,
) -> str:
    """Build a prompt from validated parser rows and deterministic facts."""
    data = validate_parser_dataframe(dataframe, target_gene=target_gene)
    facts = compute_dataset_facts(data, target_gene=target_gene)
    chains = chains or load_prompt_chains()
    return chains["narrate"].format(
        target_gene=facts.target_name,
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
    if (
        summary.highest_mean_salinity.strip().casefold()
        != facts.highest_mean_salinity.casefold()
    ):
        errors.append(
            f"highest_mean_salinity={summary.highest_mean_salinity!r}, "
            f"expected {facts.highest_mean_salinity!r}"
        )
    if not math.isclose(
        float(summary.highest_mean_fold_change_rq),
        facts.highest_mean_fold_change_rq,
        rel_tol=0,
        abs_tol=PEAK_RQ_ABS_TOLERANCE,
    ):
        errors.append(
            "highest_mean_fold_change_rq="
            f"{summary.highest_mean_fold_change_rq}, "
            f"expected {facts.highest_mean_fold_change_rq}"
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
    summary.headline = (
        f"{facts.target_name} had the highest mean measured fold change in the "
        f"{facts.highest_mean_salinity} condition "
        f"({facts.highest_mean_fold_change_rq:.3f})."
    )
    summary.peak_condition = (
        f"The strongest measured {facts.target_name} response was "
        f"{facts.peak_fold_change_rq:.3f} in the {facts.peak_salinity} condition "
        f"at time point {_display_number(facts.peak_time_point)}."
    )
    summary.highest_mean_salinity = facts.highest_mean_salinity
    summary.highest_mean_fold_change_rq = facts.highest_mean_fold_change_rq
    condition_means = _condition_means_text(facts)
    if len(facts.time_points) == 1:
        summary.trend_description = (
            "Only one measured time point "
            f"({_display_number(facts.time_points[0])}) is represented, so no "
            "time-course trend can be inferred. Mean measured fold change by "
            f"condition: {condition_means}."
        )
    else:
        summary.trend_description = (
            "Mean measured fold change across accepted rows by condition: "
            f"{condition_means}. Multiple time points are represented; this "
            "descriptive summary does not perform time-course modeling."
        )
    summary.key_observations = [
        f"The highest condition mean was {facts.highest_mean_fold_change_rq:.3f} "
        f"in {facts.highest_mean_salinity}.",
        f"The strongest individual measurement was "
        f"{facts.peak_fold_change_rq:.3f} in {facts.peak_salinity} at time point "
        f"{_display_number(facts.peak_time_point)}.",
    ]
    if len(facts.time_points) == 1:
        summary.key_observations.append(
            "Only one measured time point is available."
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
    target_gene: str = DEFAULT_TARGET_GENE,
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
    attempt_errors: list[str] = []
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
            summary._generated_text = summary.as_text()
            return _normalize_deterministic_fields(summary, facts)
        except (ValidationError, ValueError) as exc:
            last_error = exc
            attempt_errors.append(f"attempt {attempt}: {exc}")
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
        "Model returned invalid or factually inconsistent output on all "
        f"{MAX_ATTEMPTS} attempts. " + " | ".join(attempt_errors)
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


def _rate_consistency(
    *,
    success_rate: float,
    successful_runs: int,
    mean_similarity: Optional[float],
    peak_agreement: Optional[float],
) -> tuple[str, list[str]]:
    """Assign a generation-consistency rating with explicit reasons."""
    reasons: list[str] = []
    if success_rate < 0.8:
        reasons.append(
            f"Only {success_rate:.0%} of requested runs passed validation; "
            "at least 80% is required for MODERATE or HIGH."
        )
    if successful_runs < 2:
        reasons.append(
            "Fewer than two runs passed, so cross-run similarity is unavailable."
        )
    if peak_agreement is not None and peak_agreement < 1.0:
        reasons.append("Validated runs did not fully agree on the measured peak.")
    if mean_similarity is not None and mean_similarity < 0.65:
        reasons.append(
            f"Mean wording similarity was {mean_similarity:.3f}, below 0.650."
        )

    if reasons:
        return "low", reasons
    if (
        success_rate == 1.0
        and mean_similarity is not None
        and mean_similarity >= 0.85
    ):
        return "high", [
            "All requested runs passed validation, peak agreement was complete, "
            "and mean wording similarity was at least 0.850."
        ]
    return "moderate", [
        "At least 80% of runs passed validation with complete peak agreement, "
        "but the HIGH threshold was not met."
    ]


def check_consistency(
    dataframe: pd.DataFrame,
    target_gene: str = DEFAULT_TARGET_GENE,
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
    failure_details: list[dict[str, Any]] = []

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
            failure_details.append(
                {
                    "run": run,
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                }
            )
            logger.error("Consistency run %d failed: %s", run, exc)

    texts = [summary.consistency_text() for summary in summaries]
    scores = _pairwise_similarities(texts) if len(texts) >= 2 else []
    if scores:
        mean_similarity: Optional[float] = statistics.mean(scores)
        stdev_similarity: Optional[float] = (
            statistics.stdev(scores) if len(scores) > 1 else 0.0
        )
        min_similarity: Optional[float] = min(scores)
        max_similarity: Optional[float] = max(scores)
    else:
        mean_similarity = None
        stdev_similarity = None
        min_similarity = None
        max_similarity = None

    if len(summaries) >= 2:
        peak_keys = [summary.peak_key() for summary in summaries]
        most_common_peak = max(set(peak_keys), key=peak_keys.count)
        peak_agreement: Optional[float] = (
            peak_keys.count(most_common_peak) / len(peak_keys)
        )
    else:
        peak_agreement = None

    success_rate = len(summaries) / runs
    rating, rating_reasons = _rate_consistency(
        success_rate=success_rate,
        successful_runs=len(summaries),
        mean_similarity=mean_similarity,
        peak_agreement=peak_agreement,
    )

    report: dict[str, Any] = {
        "input_schema": PARSER_SCHEMA_COLUMNS,
        "dataset_facts": facts.model_dump(),
        "runs_requested": runs,
        "runs_succeeded": len(summaries),
        "runs_failed": len(failure_details),
        "success_rate": round(success_rate, 4),
        "failure_details": failure_details,
        "mean_similarity": (
            round(mean_similarity, 4) if mean_similarity is not None else None
        ),
        "stdev_similarity": (
            round(stdev_similarity, 4) if stdev_similarity is not None else None
        ),
        "min_similarity": (
            round(min_similarity, 4) if min_similarity is not None else None
        ),
        "max_similarity": (
            round(max_similarity, 4) if max_similarity is not None else None
        ),
        "reliability_rating": rating,
        "rating_reasons": rating_reasons,
        "reliability_scope": (
            "Validated generation consistency only; this rating incorporates "
            "validation success rate, peak agreement, and wording similarity. "
            "It is not statistical confidence or biological validation."
        ),
        "peak_condition_agreement": (
            round(peak_agreement, 4) if peak_agreement is not None else None
        ),
        "summaries": [summary.model_dump() for summary in summaries],
    }
    logger.info(
        "Consistency: success_rate=%.3f mean=%s peak_agreement=%s rating=%s",
        success_rate,
        f"{mean_similarity:.3f}" if mean_similarity is not None else "n/a",
        f"{peak_agreement:.3f}" if peak_agreement is not None else "n/a",
        rating,
    )
    return report


def run_consistency_report(
    input_file: PathLike = DEFAULT_INPUT_FILE,
    output_file: PathLike = DEFAULT_OUTPUT_FILE,
    target_gene: str = DEFAULT_TARGET_GENE,
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


def discover_parser_outputs(
    parsed_directory: PathLike = DEFAULT_PARSED_DIRECTORY,
) -> list[Path]:
    """Find accepted parser CSVs while excluding rejection reports."""
    directory = Path(parsed_directory)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Parsed-data directory not found: {directory}. Run parser.py first."
        )
    outputs = sorted(
        (
            path
            for path in directory.glob("*.csv")
            if not path.stem.endswith("_rejected")
        ),
        key=lambda path: path.name.casefold(),
    )
    if not outputs:
        raise FileNotFoundError(
            f"No accepted parser CSVs found in {directory}. Run parser.py first."
        )
    return outputs


def run_report_batch(
    parsed_directory: PathLike = DEFAULT_PARSED_DIRECTORY,
    report_directory: PathLike = DEFAULT_REPORT_DIRECTORY,
    runs: int = CONSISTENCY_RUNS,
    *,
    provider: Optional[InferenceProvider] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
) -> list[tuple[Path, Path, dict[str, Any]]]:
    """Narrate every accepted CSV produced by the no-argument parser run."""
    parsed_files = discover_parser_outputs(parsed_directory)
    report_directory = Path(report_directory)
    provider = provider or OllamaInferenceProvider()
    completed: list[tuple[Path, Path, dict[str, Any]]] = []
    for input_path in parsed_files:
        output_path = (
            report_directory / f"{input_path.stem}_consistency_report.json"
        )
        logger.info("Narrating parser output %s", input_path)
        report = run_consistency_report(
            input_file=input_path,
            output_file=output_path,
            target_gene=DEFAULT_TARGET_GENE,
            runs=runs,
            provider=provider,
            model=model,
            temperature=temperature,
        )
        completed.append((input_path, output_path, report))
    return completed


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Narrate the clean five-column CSV produced by parser.py."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        help=(
            "Parser output CSV. If omitted, narrate every accepted CSV in "
            "datasets/parsed."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "JSON report path for a single input "
            f"(default: {DEFAULT_OUTPUT_FILE})"
        ),
    )
    parser.add_argument(
        "-t",
        "--target",
        default=DEFAULT_TARGET_GENE,
        help="Target gene to narrate (default: detect the CSV's only target)",
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
    parser.add_argument(
        "--parsed-dir",
        default=str(DEFAULT_PARSED_DIRECTORY),
        help=(
            "Parsed CSV directory used when no input filename is supplied "
            f"(default: {DEFAULT_PARSED_DIRECTORY})"
        ),
    )
    parser.add_argument(
        "--report-dir",
        default=str(DEFAULT_REPORT_DIRECTORY),
        help=(
            "Batch JSON report directory "
            f"(default: {DEFAULT_REPORT_DIRECTORY})"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point."""
    arguments = _build_cli_parser().parse_args(argv)

    if arguments.input_file is None and (
        arguments.output
        or arguments.target.casefold() != DEFAULT_TARGET_GENE.casefold()
    ):
        logger.error("--output and --target require an explicit input CSV")
        return 1

    try:
        provider = OllamaInferenceProvider(host=arguments.host)
        if arguments.input_file is None:
            completed = run_report_batch(
                parsed_directory=arguments.parsed_dir,
                report_directory=arguments.report_dir,
                runs=arguments.runs,
                provider=provider,
                model=arguments.model,
                temperature=arguments.temperature,
            )
            print("\nASTRA narrator batch complete")
            for input_path, output_path, report in completed:
                print(
                    f"- {input_path.name}: "
                    f"{report['runs_succeeded']}/{report['runs_requested']} "
                    f"successful run(s), "
                    f"{report['reliability_rating'].upper()} -> {output_path}"
                )
            return 0

        output_file = Path(arguments.output or DEFAULT_OUTPUT_FILE)
        report = run_consistency_report(
            input_file=arguments.input_file,
            output_file=output_file,
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
    print(f"Report: {output_file}")
    print("Rating reason(s):")
    for reason in report["rating_reasons"]:
        print(f"- {reason}")
    if report["summaries"]:
        print("\nValidated sample summary:")
        print(json.dumps(report["summaries"][0], indent=2))
    else:
        print("\nNo model output passed validation; see failure_details in the report.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
