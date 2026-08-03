# SmartCare Hospital AI — Disease Risk Classification System

A machine learning system that classifies hospital patients into disease risk
categories (Low, Medium, High) using clinical, demographic, and operational
data. Built to support preventive care planning and clinical decision-making
through interpretable, evaluated, and deployable predictive models.

This repository contains the full pipeline: data preprocessing, exploratory
analysis, model training and comparison, evaluation, explainability, and a
deployed prediction prototype.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
- [Pipeline Execution](#pipeline-execution)
- [Model Development](#model-development)
- [Evaluation](#evaluation)
- [Explainability](#explainability)
- [Prototype Application](#prototype-application)
- [Branching Strategy](#branching-strategy)
- [Contribution Workflow](#contribution-workflow)
- [Team](#team)
- [Reports and Documentation](#reports-and-documentation)
- [License](#license)

---

## Overview

Healthcare providers generate large volumes of patient, clinical, and
operational data, but extracting actionable insight from it at scale requires
systematic, well-validated modeling rather than ad hoc analysis. This project
implements a complete, reproducible machine learning workflow — from raw data
to a deployed prediction interface — to classify patient disease risk level
based on demographic, clinical, and operational features.

The system is designed with the same rigor expected in a production ML
project: modular, testable source code separated from exploratory notebooks;
a documented and reproducible preprocessing pipeline; multiple models trained
and compared under consistent evaluation criteria; and model interpretability
built in from the start rather than added as an afterthought.

## Problem Statement

Early identification of patient disease risk enables preventive intervention,
better resource allocation, and improved patient outcomes. Manual risk
assessment does not scale with hospital data volume and is subject to
inconsistency across clinicians.

This project addresses that gap by developing a supervised classification
model that predicts a patient's disease risk level — Low, Medium, or High —
from available clinical and demographic attributes, with transparent,
explainable output suitable for clinical decision support.

**Task type:** Multi-class classification
**Target variable:** `disease_risk_level`
**Classes:** `Low`, `Medium`, `High`

## Dataset

The system uses the SmartCare Hospital AI Dataset, comprising 1,000 patient
records across four domains:

| Domain | Attributes |
|---|---|
| Patient Information | Patient ID, Age, Gender, Blood Group |
| Clinical Information | Diagnosis, Blood Pressure, Blood Sugar, Cholesterol, BMI |
| Hospital Operations | Department, Appointment History, Previous Admissions, Length of Stay, Room Type, Treatment Count, Laboratory Test Count |
| Financial Data | Consultation Charges, Laboratory Charges, Room Charges, Medicine Charges, Total Bill Amount |

Raw data and the accompanying data dictionary are not committed to version
control and must be placed locally under `data/raw/` (see
[Getting Started](#getting-started)).

## System Architecture

The pipeline is organized as a sequence of independently owned, testable
stages, each consuming the previous stage's output and producing a defined
artifact for the next:

```
Raw Data
   |
   v
Preprocessing & Feature Engineering  ->  data/processed/*.csv
   |
   v
Exploratory Data Analysis            ->  reports/figures/eda/*.png
   |
   v
Model Development                    ->  models/*.pkl
   |
   v
Model Evaluation                     ->  reports/model_comparison_table.csv
   |                                     models/best_model.pkl
   v
Explainable AI Analysis              ->  reports/figures/shap/*.png
   |
   v
Prediction Prototype (Streamlit)     ->  interactive risk assessment
```

Each stage is implemented as an independent, importable Python module under
`src/`, decoupled from any single notebook. Notebooks act as thin execution
and reporting layers over this shared codebase, ensuring the same logic
that produces the reported results is what runs in the prototype.

## Repository Structure

```
smartcare-ai-project/
├── data/
│   ├── raw/                    Original dataset and data dictionary (not versioned)
│   └── processed/              Cleaned and feature-engineered data (generated)
├── notebooks/                  Analysis and reporting notebooks, one per pipeline stage
├── src/                        Core, reusable pipeline modules
│   ├── config.py               Centralized paths and configuration
│   ├── preprocessing.py        Data cleaning and preprocessing
│   ├── feature_engineering.py  Feature construction and selection utilities
│   ├── eda_utils.py            Exploratory analysis and visualization functions
│   ├── train_models.py         Model training and hyperparameter tuning
│   ├── evaluate.py             Model evaluation and comparison
│   ├── explainability.py       SHAP-based model interpretation
│   └── utils.py                Shared helper functions
├── models/                     Trained model artifacts and metadata
├── reports/                    Generated figures, comparison tables, technical report
├── app/                        Streamlit prediction prototype
├── literature_review/          Supporting research references
├── presentation/               Presentation materials
└── docs/                       Contribution log and internal documentation
```

## Technology Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Prototype / Deployment | Streamlit |
| Development Environment | Jupyter Notebook |
| Version Control | Git, GitHub |

## Getting Started

### Prerequisites

- Python 3.10 or later
- pip

### Installation

```bash
git clone https://github.com/Thrithwaka/smartcare-ai-project.git
cd smartcare-ai-project

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Data Setup

Place the following files, provided separately, into `data/raw/`:

- `smartcare_ai_dataset_1000.csv`
- `smartcare_ai_dataset_data_dictionary.csv`

## Pipeline Execution

Run each stage independently:

```bash
python -m src.preprocessing
python -m src.train_models
python -m src.evaluate
python -m src.explainability
```

Or execute the full pipeline end-to-end via notebook:

```bash
jupyter notebook notebooks/06_final_pipeline_demo.ipynb
```

## Model Development

Three classification models are trained and compared under consistent
cross-validation and hyperparameter search:

- Logistic Regression — interpretable linear baseline
- Random Forest — non-linear ensemble, robust to mixed feature types
- XGBoost — gradient-boosted ensemble, typically the strongest baseline

Each model is tuned via grid search optimizing macro-averaged F1 score,
selected over accuracy to account for potential class imbalance across
the three risk categories.

## Evaluation

Models are evaluated on a held-out, stratified test set using:

- Accuracy
- Precision (macro-averaged)
- Recall (macro-averaged)
- F1 Score (macro-averaged)
- Confusion Matrix

The highest macro-F1 model is selected as the production model and
persisted to `models/best_model.pkl`. Full comparison results are written
to `reports/model_comparison_table.csv`.

## Explainability

Model predictions are interpreted using SHAP (SHapley Additive exPlanations),
providing both:

- Global explanations — which features most influence predictions overall
- Local explanations — why a specific patient received a specific
  risk classification

This supports transparency and clinical trust in model output, and is
surfaced directly in the prediction prototype.

## Prototype Application

An interactive Streamlit application allows a user to input patient
attributes and receive a real-time risk classification, along with the
top contributing factors behind the prediction.

```bash
streamlit run app/app.py
```

## Branching Strategy

This repository follows a structured, task-ownership branching model:

| Branch | Purpose |
|---|---|
| `main` | Stable, submission-ready state. Protected, merged only via reviewed pull request. |
| `develop` | Integration branch where completed work is combined and validated before release to `main`. |
| `chore/shared` | Shared, non-task-specific changes: documentation, configuration, dependencies. |
| `feature/<owner>-<task>` | Individually owned pipeline stage development. |

Merges into `develop` follow pipeline dependency order, since each stage
consumes the previous stage's output.

## Contribution Workflow

1. Create or check out your assigned feature branch.
2. Pull the latest `develop` before starting new work.
3. Commit incrementally with clear, descriptive messages.
4. Open a pull request into `develop`; require at least one review before merging.
5. After full pipeline validation on `develop`, release to `main` via a
   final reviewed pull request and tag the submission commit.

## Team

| Contributor | Responsibility |
|---|---|
| [Chanuu](https://github.com/<add-username>) | Data preprocessing and feature engineering |
| [Avishka](https://github.com/deshanavishka125-dot) | Exploratory data analysis |
| [Thrithwaka](https://github.com/Thrithwaka) | Model development and hyperparameter tuning |
| [Ramda](https://github.com/<add-username>) | Model evaluation and selection |
| [Tharindu](https://github.com/nvtharindukothalawala-tech) | Explainable AI analysis and prototype development |

Detailed individual contribution records are maintained in
`docs/individual_contributions.md`.

## Results

Model performance is evaluated on a held-out, stratified test set using
macro-averaged precision, recall, and F1 score across all three risk
classes. Full, up-to-date results are generated by the evaluation stage
and written to `reports/model_comparison_table.csv`; summary figures are
available under `reports/figures/`.

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| Logistic Regression | — | — | — | — |
| Random Forest | — | — | — | — |
| XGBoost | — | — | — | — |

The model with the highest macro F1 score is selected as the production
model (`models/best_model.pkl`). Regenerate this table by running
`python -m src.evaluate`.

## Documentation

Further documentation is maintained under `docs/`, including design
notes, the internal contribution log, and reviewer guidance. Research
references supporting the modeling approach are collected in
`literature_review/references.bib`.

## Roadmap

Planned improvements beyond the current scope:

- Hyperparameter optimization via Bayesian search
- Ensemble/stacking across the trained models
- Deployment of the prototype to a managed cloud environment
- Extension to additional prediction tasks (appointment no-show, readmission)

## Acknowledgments

Built using open-source tooling from the scikit-learn, XGBoost, SHAP, and
Streamlit communities.

## License

This project is licensed under the terms described in the `LICENSE` file.