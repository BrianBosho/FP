## Reports (curated artifacts)

This folder is for **intentional, curated** outputs you want to version control:

- Paper-ready figures (e.g., `figures/*.png`, `figures/*.pdf`)
- Final tables (e.g., `tables/*.csv`)
- Notes describing how a figure/table was produced

### What does NOT belong here

Generated run outputs should go under `runs/` (gitignored), including:

- Experiment logs / raw result dumps
- Intermediate plots
- Per-run CSVs/JSONs
- W&B artifacts

### Suggested structure

- `reports/figures/`
- `reports/tables/`
- `reports/notes/`

