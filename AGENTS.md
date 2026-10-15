# Repository Guidelines

## Project Structure & Module Organization

This repository is a research workspace for agentic search. The Python project lives in `agentic-search/`; shared benchmark files live under `data/`, and `literature/` is supporting material excluded from Git:

- `literature/papers.md` contains reading cards and research leads.
- `literature/papers.csv` is the filterable inventory.
- `literature/references.bib` contains provisional citations.
- `literature/AI搜索与Agentic搜索_2026顶会进展.md` is the conference survey.
- `literature/outputs/agentic-search-20261009/` contains the reading tracker and export.
- `agentic-search/pyproject.toml` defines the research package and its dependencies.
- `agentic-search/src/agentic_search/` contains research implementation code.
- `agentic-search/benchmark/` contains evaluation protocols, entry points, and benchmark-specific configuration.
- `data/browsecomp_plus/README.md` documents the pinned dataset revisions and local outputs.
- `.agent/.env` holds local evaluation credentials and is ignored by Git.
- `third_party/BrowseComp-Plus/` is the upstream evaluation repository; treat it as a pinned vendor checkout (plain clone, not a submodule) and do not edit it directly.

Within `agentic-search/`, keep code in `src/agentic_search/`, benchmark evaluation in `benchmark/`, other entry points in `experiments/`, configuration in `configs/`, tests in `tests/`, and generated results in `results/`. Keep large datasets, checkpoints, and generated results out of Git.

## Build, Test, and Development Commands

The research package uses Python 3.10+ and Hatchling. Run these commands from the outer repository root:

```bash
uv sync --project agentic-search
uv run --project agentic-search python -c "import agentic_search"
uv build --project agentic-search
```

These commands install the package, verify import, and build distributions, respectively. Add dependencies with `uv add --project agentic-search <package>`. Benchmark entry points should be run with `uv run --project agentic-search python agentic-search/benchmark/scripts/<entrypoint>.py`. There is no experiment runner or test suite in this package yet. As code is added, document a smoke run and the full experiment. Record dataset, model, seed, and evaluation settings with each result.

## Research and Reproducibility

Every experiment should state its question or hypothesis, baseline, split, metrics, and expected artifact. Save configuration and logs with results. Separate exploratory analysis from final numbers and explain protocol changes. Literature notes motivate claims but do not replace source verification.

## Coding Style & Naming Conventions

Use UTF-8 and concise Markdown with ATX headings. Prefer descriptive lowercase names for new code and artifacts (for example, `experiments/trajectory_ablation.py` and `configs/baseline.yaml`). Use ISO dates (`YYYY-MM-DD`) in dated outputs. Keep CSV columns stable and follow the existing BibTeX key pattern, such as `agentic_s1_2026`.

## Testing Guidelines

There is no coverage requirement yet. At minimum, new data-processing code should have a focused test for parsing and one representative end-to-end smoke test. Before sharing literature updates, confirm that Markdown links work, CSV rows parse, and the Markdown, CSV, BibTeX, and workbook agree on paper IDs and titles.

## Commit & Pull Request Guidelines

This directory has no Git history, so no repository-specific convention can be inferred. If version control is added, use concise imperative subjects, such as `research: add trajectory baseline`. Pull requests should describe the research question, assumptions changed, commands run, affected datasets, and result location; include plots or screenshots when useful.

## Data Integrity & Review Notes

Preserve source URLs and distinguish screening notes (`初筛`) from verified metadata. Do not silently overwrite reading assignments or experimental results. When correcting bibliographic metadata, update all corresponding inventory exports together.
