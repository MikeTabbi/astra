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
            "fold_change_rq": [4.0, 18.0, 3.5],
            "variance_sd": [0.4, 0.2, 0.3],
        }
    )


def _valid_summary(**overrides: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "target_name": "C5",
        "headline": "C5 expression was highest in the Low condition.",
        "trend_description": (
            "At time point 24, measured expression differed across the three "
            "categorical conditions."
        ),
        "peak_condition": (
            "The strongest measured response was in Low at time point 24."
        ),
        "peak_time_point": 24,
        "peak_salinity": "Low",
        "peak_fold_change_rq": 18.0,
        "confidence_note": (
            "This small descriptive dataset does not establish significance "
            "or causality."
        ),
        "key_observations": [
            "The Low condition had the maximum measured fold change.",
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

        self.assertEqual(arguments.input_file, "parsed_output.csv")
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
        self.assertEqual(result.peak_fold_change_rq, 18.0)

    def test_consistency_report_uses_structured_peak_agreement(self) -> None:
        provider = FakeProvider([_valid_summary() for _ in range(5)])

        report = astra_narrator.check_consistency(
            _parser_output(),
            provider=provider,
        )

        self.assertEqual(report["runs_succeeded"], 5)
        self.assertEqual(report["peak_condition_agreement"], 1.0)
        self.assertEqual(report["reliability_rating"], "high")
        self.assertIn("not statistical confidence", report["reliability_scope"])

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


if __name__ == "__main__":
    unittest.main()
