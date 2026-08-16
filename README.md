# SmartCare AI

## AI-Powered Disease Risk Classification System

SmartCare AI is a group-developed machine learning system for classifying hospital patients into three disease-risk categories: **Low, Medium, and High**.

The system combines patient demographic, clinical, hospital-administrative, and healthcare-related information to generate an ML-based risk prediction. The project is designed as an end-to-end machine learning workflow covering data understanding, preprocessing, exploratory data analysis, model development, hyperparameter optimization, ensemble learning, model evaluation, explainable AI, and an interactive Streamlit prototype.

> **Important:** SmartCare AI is a research and decision-support prototype. It is not a medical diagnostic system and must not be used as a substitute for qualified clinical judgement.

---

## Table of Contents

- [Overview](#overview)
- [Project Objectives](#project-objectives)
- [Problem Definition](#problem-definition)
- [Dataset](#dataset)
- [System Architecture](#system-architecture)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Model Development](#model-development)
- [Model Evaluation](#model-evaluation)
- [Explainable AI](#explainable-ai)
- [Prediction Prototype](#prediction-prototype)
- [Technology Stack](#technology-stack)
- [System Requirements](#system-requirements)
- [Python and Package Versions](#python-and-package-versions)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Running the Project](#running-the-project)
- [Expected Outputs](#expected-outputs)
- [Model and Data Artifacts](#model-and-data-artifacts)
- [Reproducibility](#reproducibility)
- [Team and Responsibilities](#team-and-responsibilities)
- [Git Workflow](#git-workflow)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Overview

Healthcare datasets contain a combination of demographic, clinical, administrative, and operational information that can be used to identify patterns associated with patient risk.

SmartCare AI applies supervised machine learning to classify patient records into:

- **Low**
- **Medium**
- **High**

risk categories.

The project follows a structured ML workflow so that preprocessing, feature engineering, model training, evaluation, explainability, and prediction use consistent artifacts and logic.

The current pipeline contains:

1. Dataset understanding
2. Data preprocessing and feature engineering
3. Exploratory data analysis
4. Machine learning model development
5. Hyperparameter optimization
6. Ensemble learning
7. Model evaluation and selection
8. SHAP and LIME explainability
9. Interactive Streamlit prediction prototype

---

## Project Objectives

The main objectives of SmartCare AI are to:

- Prepare a reliable dataset for machine learning.
- Identify important patterns and relationships in patient data.
- Engineer meaningful features for risk classification.
- Train and compare multiple classification models.
- Optimize model hyperparameters using cross-validation.
- Investigate ensemble learning as an advanced modeling technique.
- Evaluate models using appropriate multi-class classification metrics.
- Identify the best-performing model based on Macro F1.
- Explain model predictions using SHAP and LIME.
- Provide an interactive prototype that accepts patient information and generates a model-based risk prediction.
- Maintain a reproducible and modular project structure.

---

## Problem Definition

### Problem Type

**Multi-class classification**

### Target Variable

```text
disease_risk_level
```

### Target Classes

```text
Low
Medium
High
```

The model learns patterns from the available patient attributes and predicts the corresponding risk category.

The prediction is intended to support analysis and decision-making workflows. It does not establish medical diagnosis or causation.

---

## Dataset

The SmartCare AI dataset contains **1,000 patient records** and includes information from several categories.

### Major Data Domains

| Domain | Example Attributes |
|---|---|
| Demographic | Age, Gender, Blood Group |
| Clinical | Diagnosis, Systolic BP, Diastolic BP, Blood Sugar, Cholesterol, BMI |
| Hospital / Admission | Admitted status, Length of Stay, Room Type, Previous Admissions, Previous Appointments |
| Administrative | Department, Appointment Status |
| Financial / Payment | Payment Method, Payment Status |
| Target | Disease Risk Level |

The model-development pipeline produces a final model-ready representation containing **66 features** after preprocessing and feature engineering.

### Data Privacy

Raw datasets are not intended to be committed to the public repository unless their redistribution is explicitly permitted.

Place locally required source data under:

```text
data/raw/
```

---

## System Architecture

```text
                    SmartCare AI
                         |
                         v
                Raw Patient Dataset
                         |
                         v
          Data Preprocessing & Cleaning
                         |
                         v
            Feature Engineering
                         |
                         v
                  66 ML Features
                         |
              +----------+----------+
              |                     |
              v                     v
        Exploratory Data       Train / Test
           Analysis              Split
                                  |
                                  v
                       Model Development
                                  |
                  +---------------+---------------+
                  |               |               |
                  v               v               v
             Logistic         Decision        Random
            Regression          Tree          Forest
                                  |
                                  v
                              XGBoost
                                  |
                                  v
                    Hyperparameter Optimization
                                  |
                                  v
                       Ensemble Learning
                                  |
                                  v
                         Model Evaluation
                                  |
                                  v
                       Final Model Selection
                                  |
                                  v
                         Logistic Regression
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
             SHAP + LIME                  Streamlit App
             Explainability               Prediction UI
                    |                           |
                    +-------------+-------------+
                                  |
                                  v
                         Risk Prediction
                     Low / Medium / High
```

---

## Machine Learning Pipeline

### Task 02 — Dataset Understanding

The dataset is examined to understand:

- Dataset dimensions
- Feature types
- Target variable
- Missing values
- Duplicate records
- Data distributions
- Class distribution
- Data quality issues

### Task 03 — Preprocessing and Feature Engineering

The preprocessing stage prepares the dataset for model training.

The workflow includes:

- Data cleaning
- Missing-value handling
- Duplicate handling
- Categorical encoding
- Feature engineering
- Feature selection
- Train/test splitting
- Numerical feature scaling
- Saving preprocessing artifacts

The model-ready dataset currently contains **66 features**.

Important generated artifacts include:

```text
data/processed/X_train.csv
data/processed/X_test.csv
data/processed/y_train.csv
data/processed/y_test.csv
models/scaler.pkl
models/feature_columns.pkl
```

### Task 04 — Exploratory Data Analysis

EDA is used to understand the structure and behaviour of the processed analytical data.

The analysis includes:

- Descriptive statistics
- Correlation analysis
- Distribution analysis
- Pattern discovery
- Histograms
- Boxplots
- Scatterplots
- Correlation heatmaps
- Class-distribution visualizations
- Interpretation of observed patterns

EDA findings are used to inform subsequent model development and interpretation.

### Task 05 — Model Development

The project trains and compares multiple classification models:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

The workflow includes:

- Baseline comparison
- Hyperparameter selection
- Cross-validation
- Macro F1 optimization
- Model comparison
- Ensemble learning

The project uses Macro F1 as an important model-selection metric because it evaluates performance across classes without allowing the majority class to dominate the score.

### Task 06 — Model Evaluation

The trained models are evaluated on the held-out test set.

The evaluation includes:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Macro ROC-AUC
- Confusion Matrix
- Model comparison
- Final model identification

The current evaluation identifies:

```text
Best Model: Logistic Regression
Macro F1:   0.9140
Accuracy:   0.9200
Macro Precision: 0.9406
Macro Recall:    0.8944
Macro ROC-AUC:   0.9914
```

These values describe performance on the project's current dataset and test split. They should not be interpreted as evidence of clinical validity or guaranteed performance on external populations.

---

## Explainable AI

SmartCare AI uses both **SHAP** and **LIME** to investigate model predictions.

### SHAP Analysis

The project includes:

- Global feature importance
- Class-specific feature importance
- Local explanation of a correctly classified instance
- Local explanation of a misclassified instance
- Feature contribution analysis

The current global SHAP analysis identified the following top features by mean absolute SHAP value:

| Feature | Mean Absolute SHAP |
|---|---:|
| `blood_sugar_mg_dl` | 4.1121 |
| `cholesterol_mg_dl` | 3.7737 |
| `bmi` | 3.3995 |
| `age` | 3.3306 |
| `previous_admissions` | 2.2922 |

### LIME Analysis

LIME is used for local, model-agnostic explanations of individual predictions.

### SHAP + LIME Comparison

The project compares important features identified by SHAP and LIME for:

- Correct predictions
- Misclassified predictions

The comparison helps investigate agreement and differences between two explainability approaches.

### Explainability and Ethics

SHAP and LIME describe the behaviour of the trained model. Their outputs should not be interpreted as proof that a feature causes a medical condition.

The explanations are intended to support transparency, human review, and model error analysis.

---

## Prediction Prototype

The project includes an interactive **Streamlit** prototype.

The prototype:

1. Accepts patient information from the user.
2. Validates the supplied values.
3. Applies the project's preprocessing and feature-engineering logic.
4. Converts the information into the expected model feature representation.
5. Loads the trained model and preprocessing artifacts.
6. Generates a risk prediction.
7. Displays the predicted risk category.
8. Displays prediction probabilities where available.
9. Provides model-based explanation information.
10. Clearly communicates that the system is a prototype and not a medical diagnosis tool.

Run the application with:

```bash
streamlit run app/streamlit_app.py
```

---

## Technology Stack

| Category | Technology | Version |
|---|---|---|
| Programming Language | Python | 3.10.x |
| Data Processing | NumPy | 1.26.4 |
| Data Processing | Pandas | 2.2.2 |
| Scientific Computing | SciPy | 1.15.3 |
| Machine Learning | Scikit-learn | 1.7.2 |
| Gradient Boosting | XGBoost | 3.0.2 |
| Explainable AI | SHAP | 0.45.0 |
| Explainable AI | LIME | 0.2.0.1 |
| Visualization | Matplotlib | 3.8.4 |
| Visualization | Seaborn | 0.13.2 |
| Visualization | Plotly | 6.9.0 |
| Prototype | Streamlit | 1.33.0 |
| Model Persistence | Joblib | 1.4.0 |
| Notebook Environment | Jupyter | 1.0.0 |
| Notebook Environment | Notebook | 7.1.3 |
| Python Kernel | ipykernel | 6.29.5 |

All required dependencies are pinned in:

```text
requirements.txt
```

---

## System Requirements

### Minimum Software Requirements

- **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS
- **Python:** 3.10.x
- **pip:** Compatible with Python 3.10
- **Git:** Required for cloning the repository
- **Web Browser:** Chrome, Edge, Firefox, or another modern browser
- **Virtual Environment:** Recommended

### Recommended Hardware

| Component | Minimum | Recommended |
|---|---:|---:|
| CPU | 2 cores | 4+ cores |
| RAM | 4 GB | 8 GB or more |
| Free Storage | 2 GB | 5 GB or more |
| Internet | Required for initial installation | Stable broadband connection |

The model-development notebooks may require more time on lower-specification systems, particularly during hyperparameter optimization and explainability analysis.

---

## Python Version

This project is developed and tested with:

```text
Python 3.10.x
```

Recommended Python version:

```text
Python 3.10.11
```

Check your installed Python version:

```bash
python --version
```

or:

```bash
python3 --version
```

Using the same Python major/minor version as the development environment is recommended for reproducibility.

---

## Repository Structure

```text
smartcare-ai-project/
│
├── app/
│   ├── streamlit_app.py
│   ├── utils/
│   │   ├── preprocessing.py
│   │   ├── prediction.py
│   │   └── explainability.py
│   └── assets/
│       └── style.css
│
├── data/
│   ├── raw/
│   │   └── dataset files
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── models/
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   └── trained model artifacts
│
├── notebooks/
│   ├── Task 02 notebook
│   ├── Task 03 notebook
│   ├── Task 04 notebook
│   ├── Task 05 notebook
│   ├── Task 06 notebook
│   └── Task 07 notebook
│
├── reports/
│   ├── figures/
│   │   ├── eda/
│   │   ├── evaluation/
│   │   └── explainability/
│   └── results/
│
├── requirements.txt
├── README.md
└── LICENSE
```

The exact filenames may vary as the repository evolves, but the dependency flow between the pipeline stages should remain consistent.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Thrithwaka/smartcare-ai-project.git
```

Move into the project directory:

```bash
cd smartcare-ai-project
```

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Verify Python

```bash
python --version
```

Expected:

```text
Python 3.10.x
```

### 4. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Verify the Environment

```bash
python -c "import numpy, pandas, sklearn, xgboost, shap, lime, streamlit; print('SmartCare AI environment OK')"
```

A successful installation should print:

```text
SmartCare AI environment OK
```

---

## Data Setup

Place the required dataset files inside:

```text
data/raw/
```

Example:

```text
data/raw/
├── smartcare_ai_dataset_1000.csv
└── smartcare_ai_dataset_data_dictionary.csv
```

Raw data should remain outside version control when redistribution is restricted.

---

## Running the Project

### Run the Streamlit Prototype

From the project root:

```bash
streamlit run app/streamlit_app.py
```

Streamlit will provide a local address, normally similar to:

```text
http://localhost:8501
```

Open the displayed address in a web browser.

### Run the Notebooks

Start Jupyter:

```bash
jupyter notebook
```

Then open the required notebook from:

```text
notebooks/
```

Recommended execution order:

```text
Task 02
   ↓
Task 03
   ↓
Task 04
   ↓
Task 05
   ↓
Task 06
   ↓
Task 07
   ↓
Task 08 Prototype
```

The order is important because later stages depend on artifacts and decisions produced by earlier stages.

---

## Expected Outputs

### Task 03

Generated preprocessing artifacts include:

```text
data/processed/X_train.csv
data/processed/X_test.csv
data/processed/y_train.csv
data/processed/y_test.csv

models/scaler.pkl
models/feature_columns.pkl
```

### Task 04

EDA outputs are stored under:

```text
reports/figures/eda/
```

Typical outputs include:

```text
histograms
boxplots
scatterplots
correlation heatmaps
class distribution charts
```

### Task 05

Model-development outputs may include:

```text
trained model files
hyperparameter search results
cross-validation results
model comparison results
ensemble results
```

### Task 06

Evaluation outputs are stored under:

```text
reports/figures/evaluation/
reports/results/
```

Typical outputs include:

```text
confusion matrices
metric comparison charts
evaluation tables
final model selection metadata
```

### Task 07

Explainability outputs are stored under:

```text
reports/figures/explainability/
reports/results/
```

Typical outputs include:

```text
SHAP summary plots
class-specific SHAP plots
local SHAP explanations
misclassification explanations
LIME explanations
SHAP/LIME comparison results
```

### Task 08

The Streamlit application provides:

```text
Patient input
      ↓
Input validation
      ↓
Feature processing
      ↓
Model prediction
      ↓
Risk classification
      ↓
Prediction probabilities
      ↓
Explainability information
```

---

## Model and Data Artifacts

The prototype depends on the same preprocessing and model artifacts generated by the ML pipeline.

Important artifacts include:

```text
models/
├── scaler.pkl
├── feature_columns.pkl
└── final trained model artifact(s)
```

### Artifact Consistency

The following must remain consistent:

```text
Task 03 preprocessing
        ↓
Feature columns
        ↓
Scaler
        ↓
Task 05 trained model
        ↓
Task 06 selected model
        ↓
Task 07 explanation
        ↓
Task 08 prediction prototype
```

Do not replace the scaler, feature-column list, or final model independently without retraining and re-evaluating the pipeline.

---

## Reproducibility

The project uses fixed package versions in `requirements.txt` to reduce environment differences.

Where applicable, experiments use fixed random seeds to improve reproducibility.

For a reproducible setup:

1. Use Python 3.10.x.
2. Create a fresh virtual environment.
3. Install `requirements.txt`.
4. Use the same dataset version.
5. Execute the notebooks in the documented order.
6. Preserve the generated preprocessing and model artifacts.
7. Use the final selected model consistently in explainability and prototype stages.

---

## Team and Responsibilities

SmartCare AI is developed as a collaborative group project.

## Team and Responsibilities

SmartCare AI is developed as a collaborative group project.

| Contributor | Primary Responsibility |
|---|---|
| [Thrithwaka](https://github.com/Thrithwaka) | Data preprocessing and feature engineering |
| [Avishka](https://github.com/deshanavishka125-dot) | Exploratory data analysis |
| [Chanu](https://github.com/Anuradhi-Gunawardhana) | Machine learning model development and hyperparameter optimization |
| [Ramda](https://github.com/ramda12) | Model evaluation and model selection |
| [Tharindu](https://github.com/nvtharindukothalawala-tech) | Explainable AI analysis and prototype development |

Individual contribution details can be maintained separately in the project's documentation.

---

## Git Workflow

The repository uses Git for collaborative development.

Recommended workflow:

```text
main
  |
  +-- develop
        |
        +-- feature/<name>-<task>
        |
        +-- chore/<purpose>
```

### Recommended Workflow

```bash
git checkout develop
git pull origin develop

git checkout -b feature/<name>-<task>
```

After completing the work:

```bash
git add .
git commit -m "Add <description>"
git push origin feature/<name>-<task>
```

Changes should be reviewed before merging into the integration branch and ultimately into `main`.

---

## Limitations

The current system has several limitations:

- The dataset contains a limited number of records relative to real-world hospital datasets.
- The model is trained and evaluated on the available dataset and may not generalize to external populations.
- The system has not been clinically validated.
- Predictions depend on the quality and completeness of the supplied patient information.
- Explainability methods describe model behaviour and do not establish causal relationships.
- The prototype is intended for research and demonstration purposes rather than autonomous clinical decision-making.
- Real-world deployment would require additional privacy, security, validation, monitoring, governance, and regulatory considerations.

---

## Future Improvements

Potential future development areas include:

- External validation using independent healthcare datasets.
- Larger and more representative datasets.
- More advanced hyperparameter optimization methods.
- Advanced ensemble and stacking techniques.
- Additional prediction tasks such as readmission and appointment no-show prediction.
- Model monitoring and drift detection.
- Automated experiment tracking.
- Cloud deployment.
- Authentication and role-based access control.
- Secure handling of sensitive healthcare information.
- Continuous model evaluation and retraining.
- More advanced explainability and fairness analysis.
- Integration with healthcare information systems where appropriate governance and authorization exist.

---

## Security and Privacy

Healthcare-related data can contain sensitive information.

When working with real patient data:

- Do not commit confidential datasets to Git.
- Do not expose personally identifiable information.
- Use appropriate access controls.
- Protect model and data artifacts where required.
- Follow applicable organizational, legal, and ethical requirements.
- Use anonymized or synthetic data for public demonstrations whenever possible.

The repository should contain only data that is legally and ethically appropriate to distribute.

---

## License

This project is licensed under the terms specified in the repository's `LICENSE` file.

If a license has not yet been selected, add an appropriate license before public distribution.

---

## Acknowledgements

The project uses open-source technologies including:

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Plotly
- SHAP
- LIME
- Streamlit
- Jupyter

---

## Contact

For project-related questions, refer to the repository maintainers and contributors listed in the project documentation.

Repository:

https://github.com/Thrithwaka/smartcare-ai-project
