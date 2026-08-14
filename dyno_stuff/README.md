# Motor Efficiency Analysis

Computes electric-motor efficiency from paired dyno + MoTeC logs and plots
the result as interactive 3-D charts.

## 1. Install dependencies

```bash
pip install numpy pandas matplotlib scipy scikit-learn plotly
```

## 2. Set up your data folder

By default the script expects this layout:

```
dyno_stuff/
└── 20 March/
    ├── Dyno_Data/
    │   ├── speedramp2_pl60_manual_dhruv.csv
    │   ├── speedramp3_pl70_manual_dhruv.csv
    │   └── ...
    └── Motec_Data/
        └── CSV Export/
            ├── pl60_1.csv
            ├── pl70_1.csv
            └── ...
```

If your data lives somewhere else, either:
- Update `FOLDER` / `DATE` at the top of `motor_efficiency.py`, or
- Pass `--dyno-dir` / `--motec-dir` on the command line (see below).

## 3. Match your file lists

Near the top of `motor_efficiency.py`, edit these two lists so each dyno
file lines up with its matching MoTeC file (same position in each list):

```python
DYNO_MATCHED = ["speedramp2_pl60_manual_dhruv.csv", ...]
MOTEC_MATCHED = ["pl60_1.csv", ...]
```

Note: the lists are paired by position, so if one list is shorter, the extra
entries in the longer list are ignored.

## 4. Run it

```bash
python motor_efficiency.py
```

This will:
1. Load and align each dyno/MoTeC pair
2. Compute efficiency (mechanical output ÷ electrical input) per sample
3. Show diagnostic alignment plots for each run
4. Open two interactive 3-D plots (scatter + fitted surface) in your browser

## Useful options

| Flag | What it does |
|---|---|
| `--dyno-dir PATH` | Override the dyno CSV folder |
| `--motec-dir PATH` | Override the MoTeC CSV folder |
| `--rpm-tol N` | RPM matching tolerance (default: 5) |
| `--rolling-window N` | Smoothing window in samples (default: 3) |
| `--no-diagnostics` | Skip the per-run matplotlib alignment plots |
| `--no-show` | Don't open plots in a browser (useful on a headless machine) |
| `--save-html` | Save the two interactive plots as `efficiency_scatter.html` and `efficiency_surface.html` |

Example — run headless and just save the HTML plots:

```bash
python motor_efficiency.py --no-diagnostics --no-show --save-html
```

## Troubleshooting

- **"No runs produced data"** — check that your dyno/MoTeC filenames are
  correctly paired in `DYNO_MATCHED` / `MOTEC_MATCHED`, and that the RPM
  ranges in each pair actually overlap.
- **"WARNING: no positive DC current detected"** — that MoTeC file has no
  usable load data and is skipped automatically.
- **Efficiency values look off** — the script assumes a gear-ratio error by
  default (`offset = [1] * len(DYNO_MATCHED)` in `main()`), which scales
  torque and RPM together. Set individual entries to `0` to disable
  correction for a specific run.
