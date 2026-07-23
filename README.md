# Project ASTRA

ASTRA normalizes qPCR instrument exports into the shared table consumed by the
grid and narrator:

```text
target_name,time_point,salinity,fold_change_rq,variance_sd
```

## Parser setup

Install the project dependencies in your Python environment:

```bash
python -m pip install -r requirements.txt
```

`xlrd` is required for legacy `.xls` workbooks. `openpyxl` handles `.xlsx`.

## Parse the lab workbook

Running the parser without a filename now uses `lab_data.xls` automatically:

```bash
python parser.py
```

The QuantStudio workbook does not contain the experiment time point. In an
interactive terminal, the parser asks you to enter that numeric value. You can
also provide it directly, for example `python parser.py --time-point 1`.

By default the parser:

1. reads measurements and QC fields from `Results`;
2. reads sample conditions from `Sample Setup`;
3. joins the worksheets by `Well`;
4. keeps the requested target gene;
5. rejects `HIGHSD`, `NOAMP`, `OUTLIERRG`, `EXPFAIL`, and omit flags;
6. maps `RQ` to `fold_change_rq` and `Ct SD` to `variance_sd`;
7. uses `Biogroup Name` as the salinity/condition label;
8. aggregates surviving technical wells to sample-level rows.

Rejected rows are written to `parsed_output_rejected.csv` with source
worksheet, source row, and a specific rejection reason.

Use `--salinity` if a workbook has no setup condition, `--no-aggregate` to keep
one accepted row per well, or `--help` to see every option.

## Parse a CSV

CSV files that already contain the canonical fields can be processed directly:

```bash
python parser.py messy_lab_export.csv \
  --target C5 \
  --output parsed_output.csv
```

## Narrate the clean parser output

After `parser.py` creates `parsed_output.csv`, run:

```bash
python narrator.py
```

The narrator validates the parser's exact five-column schema, treats `High`,
`Low`, and `Control` as categorical experimental conditions, and calculates the
measured peak directly in Python. It then requests five Pydantic-structured
summaries from Ollama, rejects summaries that contradict the measured peak, and
writes `consistency_report.json`.

Set the Ollama host and model with environment variables when needed:

```bash
export OLLAMA_HOST=http://HOST_MACHINE_IP:11434
export OLLAMA_MODEL=llama3:latest
python narrator.py
```

By default, the narrator connects to Ollama on the same computer at
`http://127.0.0.1:11434`. Set `OLLAMA_HOST` only when Ollama is running on a
different team computer.

The report's reliability rating measures generation consistency only. It is not
a statistical confidence score or a substitute for biological review.

## Tests

```bash
python -m unittest discover -s tests -v
```
