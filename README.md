# Building Energy Sensor Robustness — Research Code Only

This repository contains **analysis code, experimental outputs and public-source data archives only** for an exploratory comparison of simulated sensor outages in building energy estimation.

**The research paper, manuscript LaTeX files, supplementary paper, research-paper figures and journal submission forms are intentionally NOT included here.** The author maintains the manuscript separately in private storage.

## What is included

- `code/UCI/`: UCI Appliances Energy Prediction reproducibility scripts, chronological checks, matched-exposure comparisons, grouping sensitivity and saved result tables.
- `code/SmartlivingEPC/`: hourly meter-increment analysis on four building sites, simulated sensor-loss analyses, validation scripts and selected original public data archives.
- `data/README.md` and `DATA_LICENSE_AND_ATTRIBUTION.md`: source URLs, data-placement instructions, provenance and attribution.

## Reproduction

Python 3.11+ is recommended.

```bash
python -m pip install -r code/SmartlivingEPC/requirements.txt
(cd code/UCI && python -m unittest test_v13_grouping -v)
(cd code/SmartlivingEPC && python -m unittest test_v14 -v)
```

The UCI archive may need to be obtained from the source DOI and placed according to `data/README.md`. Experimental missingness is **simulated** and not field-observed hardware failure. The two datasets have different energy targets, timescales and sensor structures; the models were retrained separately per building. UCI results are exploratory because the historical test period was examined during iterative development.

## Data credits

- UCI Appliances Energy Prediction: https://doi.org/10.24432/C5VC8G
- SmartlivingEPC Building Complex measurements: https://doi.org/10.5281/zenodo.15781077

See `DATA_LICENSE_AND_ATTRIBUTION.md` for the source-data licence and attribution. No licence is granted here for the analysis code unless separately specified by the author.
