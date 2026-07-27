"""Offline tests for the parser-to-narrator contract."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import narrator as astra_narrator  # noqa: E402


def _parser_output() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_name": ["C5", "C5", "C5"],
            "time_point": [24, 24, 24],
            "salinity": ["High", "Low", "Control"],
            "fold_change_rq": [4.0, 18.346271514892578, 3.5],
            "variance_sd": [0.4, 0.2, 0.3],
        }
    )


def _p2_output() -> pd.DataFrame:
    """Peak is High, while the highest condition mean is Low."""
    return pd.DataFrame(
        {
            "target_name": ["PHVUL.001G136100"] * 4,
            "time_point": [168] * 4,
            "salinity": ["High", "High", "Low", "Low"],
            "fold_change_rq": [100.0, 1.0, 60.0, 60.0],
            "variance_sd": [0.2, 0.2, 0.1, 0.1],
        }
    )


def _valid_summary(**overrides: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "target_name": "C5",
        "headline": (
            "C5 had the highest mean measured fold change in the Low condition."
        ),
        "trend_description": (
            "At time point 24, measured expression differed across the three "
            "categorical conditions."
        ),
        "peak_condition": (
            "The strongest measured response was in Low at time point 24."
        ),
        "peak_time_point": 24,
        "peak_salinity": "Low",
        "peak_fold_change_rq": 18.346,
        "highest_mean_salinity": "Low",
        "highest_mean_fold_change_rq": 18.346,
        "confidence_note": (
            "This small descriptive dataset does not establish significance "
            "or causality."
        ),
        "key_observations": [
            "The Low condition had the highest condition mean.",
            "Only one time point is represented.",
        ],
    }
    summary.update(overrides)
    return summary


class FakeProvider:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.calls = 0

    def generate_narration(
        self,
        *,
        prompt: str,
        schema: dict[str, Any],
        model: str,
        temperature: float,
    ) -> str:
        del prompt, schema, model, temperature
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return json.dumps(response)


class NarratorTests(unittest.TestCase):
    def test_cli_defaults_to_parser_output(self) -> None:
        arguments = astra_narrator._build_cli_parser().parse_args([])

        self.assertIsNone(arguments.input_file)
        self.assertEqual(arguments.target, "auto")
        self.assertEqual(arguments.runs, 5)

    def test_parser_contract_is_validated(self) -> None:
        data = astra_narrator.validate_parser_dataframe(_parser_output())

        self.assertEqual(
            data.columns.tolist(),
            astra_narrator.PARSER_SCHEMA_COLUMNS,
        )
        self.assertEqual(data["salinity"].tolist(), ["High", "Low", "Control"])

        with self.assertRaisesRegex(ValueError, "Missing column"):
            astra_narrator.validate_parser_dataframe(
                _parser_output().drop(columns="variance_sd")
            )

    def test_prompt_treats_salinity_as_a_category(self) -> None:
        prompt = astra_narrator.build_prompt(_parser_output())

        self.assertIn("categorical condition label", prompt)
        self.assertIn(
            "categorical salinity/condition labels: Control, High, Low",
            prompt,
        )
        self.assertNotIn("salinity in ppt", prompt)

    def test_narrate_retries_a_wrong_peak_claim(self) -> None:
        provider = FakeProvider(
            [
                _valid_summary(
                    peak_condition="The strongest response was in High.",
                    peak_salinity="High",
                ),
                _valid_summary(),
            ]
        )

        result = astra_narrator.narrate(
            _parser_output(),
            provider=provider,
        )

        self.assertEqual(provider.calls, 2)
        self.assertEqual(result.peak_salinity, "Low")
        self.assertEqual(result.peak_fold_change_rq, 18.346)

    def test_narrate_retries_unsupported_significance_claim(self) -> None:
        provider = FakeProvider(
            [
                _valid_summary(
                    headline="C5 was significantly increased in Low.",
                ),
                _valid_summary(),
            ]
        )

        result = astra_narrator.narrate(
            _parser_output(),
            provider=provider,
        )

        self.assertEqual(provider.calls, 2)
        self.assertNotIn("significant", result.headline.casefold())
        self.assertIn("Low", result.peak_condition)
        self.assertIn("no statistical significance", result.confidence_note)

    def test_auto_target_and_peak_wording_are_not_hardcoded_to_c5(self) -> None:
        data = _parser_output().assign(target_name="SB1")
        provider = FakeProvider(
            [
                _valid_summary(
                    target_name="SB1",
                    headline=(
                        "SB1 had the highest mean measured fold change in the "
                        "Low condition."
                    ),
                )
            ]
        )

        result = astra_narrator.narrate(data, provider=provider)

        self.assertEqual(result.target_name, "SB1")
        self.assertIn("measured SB1 response", result.peak_condition)
        self.assertNotIn("C5 response", result.peak_condition)

    def test_peak_and_highest_condition_mean_are_kept_separate(self) -> None:
        facts = astra_narrator.compute_dataset_facts(_p2_output())

        self.assertEqual(facts.peak_salinity, "High")
        self.assertEqual(facts.peak_fold_change_rq, 100.0)
        self.assertEqual(facts.highest_mean_salinity, "Low")
        self.assertEqual(facts.highest_mean_fold_change_rq, 60.0)

    def test_narrate_retries_an_incorrect_condition_mean_claim(self) -> None:
        valid_p2_summary = _valid_summary(
            target_name="PHVUL.001G136100",
            headline=(
                "PHVUL.001G136100 had the highest condition mean in Low."
            ),
            trend_description=(
                "Condition means were High=50.500 and Low=60.000 at time 168."
            ),
            peak_condition=(
                "The strongest individual measurement was in High at time 168."
            ),
            peak_time_point=168,
            peak_salinity="High",
            peak_fold_change_rq=100.0,
            highest_mean_salinity="Low",
            highest_mean_fold_change_rq=60.0,
            key_observations=[
                "The strongest individual measurement was in High.",
                "Low had the highest condition mean.",
            ],
        )
        provider = FakeProvider(
            [
                {
                    **valid_p2_summary,
                    "highest_mean_salinity": "High",
                    "highest_mean_fold_change_rq": 50.5,
                },
                valid_p2_summary,
            ]
        )

        result = astra_narrator.narrate(_p2_output(), provider=provider)

        self.assertEqual(provider.calls, 2)
        self.assertEqual(result.peak_salinity, "High")
        self.assertEqual(result.highest_mean_salinity, "Low")
        self.assertIn("Low condition", result.headline)

    def test_free_form_condition_prose_is_replaced_with_calculated_means(
        self,
    ) -> None:
        provider = FakeProvider(
            [
                _valid_summary(
                    headline=(
                        "C5 was higher in the High condition than the Low "
                        "condition."
                    ),
                )
            ]
        )

        result = astra_narrator.narrate(_parser_output(), provider=provider)

        self.assertEqual(provider.calls, 1)
        self.assertEqual(result.highest_mean_salinity, "Low")
        self.assertNotIn("higher", result.headline.casefold())
        self.assertIn("Control=3.500", result.trend_description)
        self.assertIn("Low=18.346", result.trend_description)

    def test_consistency_report_uses_structured_peak_agreement(self) -> None:
        provider = FakeProvider([_valid_summary() for _ in range(5)])

        report = astra_narrator.check_consistency(
            _parser_output(),
            provider=provider,
        )

        self.assertEqual(report["runs_succeeded"], 5)
        self.assertEqual(report["peak_condition_agreement"], 1.0)
        self.assertEqual(report["reliability_rating"], "high")
        self.assertEqual(report["success_rate"], 1.0)
        self.assertEqual(report["failure_details"], [])
        self.assertIn("not statistical confidence", report["reliability_scope"])

    def test_failed_runs_are_recorded_and_force_a_low_rating(self) -> None:
        invalid = _valid_summary(
            peak_condition="The strongest measured response was in High.",
            peak_salinity="High",
        )
        provider = FakeProvider(
            [invalid, invalid, invalid, invalid, invalid, invalid]
            + [_valid_summary(), _valid_summary()]
        )

        report = astra_narrator.check_consistency(
            _parser_output(),
            provider=provider,
        )

        self.assertEqual(report["runs_succeeded"], 2)
        self.assertEqual(report["runs_failed"], 3)
        self.assertEqual(report["success_rate"], 0.4)
        self.assertEqual(report["reliability_rating"], "low")
        self.assertEqual(len(report["failure_details"]), 3)
        self.assertIn("attempt 1", report["failure_details"][0]["reason"])
        self.assertIn("40%", report["rating_reasons"][0])

    def test_report_is_still_created_when_every_run_fails(self) -> None:
        invalid = _valid_summary(
            peak_condition="The strongest measured response was in High.",
            peak_salinity="High",
        )
        provider = FakeProvider([invalid])

        report = astra_narrator.check_consistency(
            _parser_output(),
            provider=provider,
        )

        self.assertEqual(report["runs_succeeded"], 0)
        self.assertEqual(report["runs_failed"], 5)
        self.assertEqual(report["success_rate"], 0.0)
        self.assertIsNone(report["mean_similarity"])
        self.assertIsNone(report["peak_condition_agreement"])
        self.assertEqual(report["reliability_rating"], "low")

    def test_report_loads_parser_csv_and_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "parsed.csv"
            output_path = Path(directory) / "report.json"
            _parser_output().to_csv(input_path, index=False)
            provider = FakeProvider([_valid_summary(), _valid_summary()])

            report = astra_narrator.run_consistency_report(
                input_path,
                output_path,
                runs=2,
                provider=provider,
            )

            saved = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(report["input_schema"], astra_narrator.PARSER_SCHEMA_COLUMNS)
            self.assertEqual(saved["dataset_facts"]["peak_salinity"], "Low")
            self.assertEqual(
                saved["dataset_facts"]["highest_mean_salinity"],
                "Low",
            )

    def test_batch_narrates_every_accepted_parser_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parsed_directory = root / "parsed"
            report_directory = root / "reports"
            parsed_directory.mkdir()
            _parser_output().to_csv(parsed_directory / "sample.csv", index=False)
            _parser_output().to_csv(
                parsed_directory / "sample_rejected.csv",
                index=False,
            )
            provider = FakeProvider([_valid_summary(), _valid_summary()])

            completed = astra_narrator.run_report_batch(
                parsed_directory,
                report_directory,
                runs=2,
                provider=provider,
            )

            self.assertEqual(len(completed), 1)
            self.assertEqual(completed[0][0].name, "sample.csv")
            self.assertTrue(
                (report_directory / "sample_consistency_report.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
