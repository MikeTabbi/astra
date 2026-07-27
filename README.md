# Project ASTRA

ASTRA converts raw qPCR exports into a five-column table and then asks a local
Ollama model to describe the cleaned measurements:

```text
target_name,time_point,salinity,fold_change_rq,variance_sd
```

## Project folders

```text
astra/
├── datasets/
│   ├── manifest.json       # datasets parser.py should process
│   ├── raw/                # original lab exports; never edited
│   ├── parsed/             # clean CSVs and rejection reports
│   ├── grids/              # pivot CSVs and color-mapped atlas JSON files
│   ├── examples/           # small test/example CSVs
│   └── synthetic/          # generator.py output
├── reports/                # narrator.py JSON reports
├── parser.py
├── grid.py
├── narrator.py
└── generator.py
```

The Python files remain in the project root so the commands stay short.

## Setup

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

`xlrd` reads the legacy `.xls` lab exports and `openpyxl` handles `.xlsx`.

## Normal workflow

From the `astra` directory, run:

```bash
python parser.py
```

With no arguments, the parser reads `datasets/manifest.json` and processes both
configured one-week USPT datasets:

- `USPT SB1 07222026_data.xls`, target `SB1`
- `USPT P2 07212026_data.xls`, target `Phvul.001G136100`

The manifest records the time point as `168` hours after stress. Each workbook
is processed separately and receives its own accepted and rejected CSV in
`datasets/parsed`.

The parser:

1. reads measurements and QC fields from `Results`;
2. reads sample conditions from `Sample Setup`;
3. joins the worksheets by `Well`;
4. keeps the configured experimental target and excludes `EF1-A`;
5. rejects `AMPNC`, `HIGHSD`, `MTP`, `NOAMP`, `OUTLIERRG`, `EXPFAIL`, and omit
   flags;
6. maps `RQ` to `fold_change_rq` and `Ct SD` to `variance_sd`;
7. uses `Biogroup Name` for High, Medium, Low, or Control;
8. aggregates surviving technical wells to sample-level rows.

Next, build the Stage 2 expression atlas:

```bash
python grid.py
```

With no arguments, the grid processes every accepted CSV in `datasets/parsed`.
For each dataset it averages `fold_change_rq` at every Salinity × Time Point
intersection and writes two files in `datasets/grids`:

- `<dataset>_grid.csv`: a standard pivot table;
- `<dataset>_atlas.json`: the same matrix with an expression classification and
  CSS color for each cell.

The default atlas uses salinity for rows, time point for columns, and a dynamic
blue-neutral-red scale around the RQ reference value of `1.0`. The axes and
calculation can be changed, for example:

```bash
python grid.py --rows time_point --columns salinity --aggregation median
```

Then run:

```bash
python narrator.py
```

With no arguments, the narrator processes every accepted CSV in
`datasets/parsed`, automatically detects its target, performs five validated
Ollama runs, and writes a separate JSON report to `reports`.

The narrator calculates both the strongest individual measurement and the
highest condition mean in Python. It validates those as separate facts, rejects
incorrect structured values, replaces free-form condition comparisons with
calculated mean-based wording, and records failed-run reasons in the JSON
report. The LOW/MODERATE/HIGH consistency rating accounts for the percentage of
runs that passed validation as well as peak agreement and original model
wording similarity.

The default Ollama model is `qwen2.5-coder:7b` and the default host is
`http://127.0.0.1:11434`.

Finally, generate traceable synthetic rows:

```bash
python generator.py
```

With no arguments, the generator processes every accepted parser CSV and
writes two files per dataset in `datasets/synthetic`:

- `<dataset>_synthetic.csv`: 1,000 reproducible synthetic rows by default;
- `<dataset>_generation_report.json`: source hash, method, seed, group-level
  comparisons, and limitations.

The generator samples real rows within each target, time point, and condition,
then applies controlled log2-scale jitter using the sampled `variance_sd`.
Every row records its source dataset, source CSV row, original RQ, applied
jitter, generation method, and random seed. It never invents a target,
condition, or time point that was not measured.

Synthetic rows are useful for pipeline and model-development experiments, but
they are not new biological evidence. With the current one-week datasets, the
generator preserves the single measured time point of 168 hours and explicitly
reports that time-series forecasting cannot yet be validated.

To change the output size or reproducible seed:

```bash
python generator.py --rows 2000 --seed 2026
```

## Add another lab dataset

1. Copy the untouched `.xls` or `.xlsx` file into `datasets/raw`.
2. Add an entry to `datasets/manifest.json` with a unique name, filename,
   experimental target, numeric time point, and output filename.
3. Run `python parser.py`, `python grid.py`, `python narrator.py`, and
   `python generator.py`.

Example manifest entry:

```json
{
  "name": "experiment_name",
  "input": "raw/export_file.xls",
  "target": "TARGET_NAME",
  "time_point": 168,
  "output": "parsed/experiment_name.csv"
}
```

## Process one file manually

The explicit single-file workflow remains available:

```bash
python parser.py "datasets/raw/USPT SB1 07222026_data.xls" \
  --target SB1 \
  --time-point 168 \
  --output datasets/parsed/sb1_manual.csv
```

```bash
python narrator.py datasets/parsed/sb1_manual.csv \
  --output reports/sb1_manual_report.json
```

```bash
python generator.py datasets/parsed/sb1_manual.csv \
  --output datasets/synthetic/sb1_manual_synthetic.csv \
  --report-output datasets/synthetic/sb1_manual_generation_report.json
```

## Tests

```bash
python -m unittest discover -s tests -v
```
