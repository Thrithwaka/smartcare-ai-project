# SmartCare Hospital AI Project

CCS3440 – Artificial Intelligence Coursework
SLTC | School of Computing & IT

## Project Overview

This project applies machine learning to the SmartCare Hospital dataset to solve one
of three prediction problems:

- **Option A** – Appointment No-Show Prediction (binary)
- **Option B** – Patient Readmission Prediction (binary)
- **Option C** – Disease Risk Classification (multi-class)

> **Selected task: Option C — Disease Risk Classification**
> **Target variable: `disease_risk_level`** → classes `{Low, Medium, High}`

## Team & Task Ownership

| Member     | Task Owned                              | Module(s) in `src/`                          |
|------------|------------------------------------------|-----------------------------------------------|
| Chanuu     | Task 03 – Preprocessing & Feature Eng.  | `preprocessing.py`, `feature_engineering.py` |
| Avishka    | Task 04 – Exploratory Data Analysis      | `eda_utils.py`                                |
| Thrithwaka | Task 05 – Model Development              | `train_models.py`                             |
| Ramda      | Task 06 – Model Evaluation                | `evaluate.py`                                 |
| Tharindu   | Task 07 & 08 – Explainable AI + Prototype | `explainability.py`, `app/app.py`             |

Each person's notebook in `notebooks/` mirrors their module and is the artifact
they should walk through in the viva.

## Project Structure

```
smartcare-ai-project/
├── data/                  Raw and processed datasets
├── notebooks/              One notebook per task/person + final demo
├── src/                    Reusable Python modules (imported by notebooks + app)
├── models/                 Saved trained models (.pkl) + metadata
├── reports/                Figures, comparison tables, technical report
├── app/                    Streamlit/Flask prototype
├── literature_review/      References for Task 01
├── presentation/           Slide deck
└── docs/                   Viva prep notes, contribution log
```

## Setup

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd smartcare-ai-project

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the provided dataset files here:
#    data/raw/smartcare_ai_dataset_1000.csv
#    data/raw/smartcare_ai_dataset_data_dictionary.csv
```

## Pipeline Order (run in this sequence)

1. `notebooks/01_preprocessing_feature_engineering.ipynb` — Chanuu
2. `notebooks/02_eda.ipynb` — Avishka
3. `notebooks/03_model_development.ipynb` — Thrithwaka
4. `notebooks/04_model_evaluation.ipynb` — Ramda
5. `notebooks/05_explainable_ai.ipynb` — Tharindu
6. `notebooks/06_final_pipeline_demo.ipynb` — combined end-to-end run (for video demo)
7. `app/app.py` — run the prototype: `streamlit run app/app.py`

## Git Workflow

- `main` branch is protected — only updated via Pull Request.
- Each person works on their own branch: `feature/<name>-<task>`
  e.g. `feature/chanuu-preprocessing`, `feature/thrithwaka-models`
- At least one teammate reviews a PR before it's merged.
- Commit early and often — grading may reference commit history as evidence
  of individual contribution.

## Reproducing Results

```bash
# Run the full pipeline end-to-end from the command line (optional helper)
python src/run_pipeline.py
```

## Deliverables Checklist (per coursework spec)

- [ ] Technical Report (PDF) — `reports/technical_report.pdf`
- [ ] Jupyter Notebooks (.ipynb) — `notebooks/`
- [ ] Python Source Code (.py) — `src/`
- [ ] Trained Model Files (.pkl) — `models/`
- [ ] Prototype Source Code — `app/`
- [ ] GitHub Repository Link
- [ ] Presentation Slides — `presentation/slides.pptx`
- [ ] Video Demonstration (5–10 min)
