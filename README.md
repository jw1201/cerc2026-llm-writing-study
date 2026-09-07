# Data and Materials: From Writing to Supervising (CERC 2026)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22113365.svg)](https://doi.org/10.5281/zenodo.22113365)

Supplementary materials for:

Weigel, J. & Müller, A. (2026). *From Writing to Supervising: A Process Study of LLM-Assisted Academic Paper Creation.* Proceedings of CERC 2026, Galway, Ireland.

## Overview

This repository documents the fully recorded process of creating an academic conference paper with GPT-5.5, including all prompts, human interventions, timing data, and the embedded sycophancy experiment that formed the content of the generated paper.

## Repository Structure

```
cerc2026-llm-writing-study/
├── README.md
├── LICENSE
├── CITATION.cff
├── data/
│   ├── raw/
│   │   ├── prompt_archive.csv
│   │   ├── process_diary.csv
│   │   ├── hallucination_log.csv
│   │   └── sycophancy_experiment/
│   │       ├── task_items.csv
│   │       └── model_responses.csv
│   └── processed/
│       ├── coding_scheme.md
│       └── cwss_scores.csv
├── code/
│   └── compute_cwss.py
└── results/
    ├── generated_sycophancy_paper.pdf
    └── figures/
```

### `data/raw/`
Unprocessed data as collected during the writing process.

- **`prompt_archive.csv`** — all prompts submitted during the writing process, with step assignment and intervention type. Columns: `Prompt_ID, Step, Prompt_Text, Intervention_Type, Timestamp`
- **`process_diary.csv`** — time tracking per writing step, distinguishing AI generation time from active human time. Columns: `Datum, Uhrzeit, Step, KI_Sek, Mensch_Sek, Gesamt_Sek, KI_Min, Mensch_Min, Gesamt_Min, KI_Taetigkeit, Mensch_Taetigkeit`
- **`hallucination_log.csv`** — all identified instances of factually incorrect output. Columns: `Case_ID, Step, Error_Type, Description, Detection_Method`
- **`sycophancy_experiment/task_items.csv`** — the 60 controlled academic judgment tasks used in the embedded sycophancy experiment (three categories, three contradiction strengths)
- **`sycophancy_experiment/model_responses.csv`** — raw model responses across both models (GPT-5.5, Claude Sonnet 4.6) and three experimental runs

### `data/processed/`
Data derived from the raw data through classification or computation.

- **`coding_scheme.md`** — the Human Intervention Taxonomy (Types A–G) applied to classify entries in `prompt_archive.csv`
- **`cwss_scores.csv`** — computed sycophancy scores (CWSS) per task, model, and run, derived from `model_responses.csv` via `code/compute_cwss.py`

### `code/`
- **`compute_cwss.py`** — script used to calculate the weighted sycophancy score (CWSS) from raw model responses

### `results/`
- **`generated_sycophancy_paper.pdf`** — the AI-generated conference paper produced during the documented writing process (the object of study, not this CERC paper itself)
- **`figures/`** — figures reproduced from the CERC 2026 paper (Figures 2 and 3)

## Methodological Context

- **Model:** GPT-5.5 Thinking, via ChatGPT desktop app (Plus subscription)
- **Period:** May 2026
- **Process framework:** twelve consecutive writing steps (S1–S12), see Figure 1 in the associated paper
- **Documentation instruments:** process diary, prompt archive, hallucination log (see `data/raw/`)

## How to Cite

If you use this data, please cite both the associated paper and this repository:

```
Weigel, J. & Müller, A. (2026). From Writing to Supervising: A Process Study of LLM-Assisted Academic Paper Creation. Proceedings of CERC 2026, Galway, Ireland.

Weigel, J. (2026). Data and Materials: From Writing to Supervising (CERC 2026) [Data set]. GitHub. https://github.com/stjiweig/cerc2026-llm-writing-study
```

See `CITATION.cff` for a machine-readable citation.

## License

Data and documentation in this repository are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Contact

jessica.b.weigel@stud.h-da.de
