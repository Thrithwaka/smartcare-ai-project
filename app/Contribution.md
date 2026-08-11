# SmartCare AI — Application and Explainable AI Contribution

**Developer:** Tharindu Kothalawala
**Component:** Streamlit Application (`app/app.py`) and Prediction / Explainability Pipeline (`app/predict.py`)
**Project:** SmartCare AI — Disease Risk Assessment Prototype (CCS3440 Coursework)

---

## Table of Contents

1. [Scope of This Contribution](#scope-of-this-contribution)
2. [Application Design (app.py)](#application-design-apppy)
3. [Prediction and Explainability Pipeline (predict.py)](#prediction-and-explainability-pipeline-predictpy)
4. [Explainable AI Approach](#explainable-ai-approach)
5. [How to Run This Component](#how-to-run-this-component)
6. [Summary](#summary)

---

## Scope of This Contribution

This document describes the two components of SmartCare AI that were independently designed and implemented as part of this contribution:

- **`app/app.py`** — the Streamlit user interface that delivers the trained model as an interactive clinical decision-support tool.
- **`app/predict.py`** — the inference pipeline that reproduces the training-time feature engineering, applies the fitted model, and generates a SHAP-based explanation for each prediction.

The trained models, dataset, and preprocessing/model-selection work carried out in the project notebooks are outside the scope of this document; this write-up covers only the application layer and the explainability layer built on top of that work.

---

## Application Design (app.py)

### Purpose

`app.py` takes the model selected during model development and turns it into a usable, non-technical clinical tool. The goal was to move beyond a notebook-only demonstration and deliver something a clinical user could realistically operate: a form for patient input, a clear risk output, and an explanation a non-data-scientist could interpret.

### Structure

The interface is organized into four sections, each implemented as an `st.container(key=...)` block styled as a distinct card:

| Card | Contents |
|---|---|
| Patient Demographics | Age, gender, blood group |
| Vitals and Lab Results | Systolic/diastolic blood pressure, blood sugar, cholesterol, BMI |
| Visit and History | Department, diagnosis, appointment status, admission status, room type, appointment and admission history, payment details |
| Results | Predicted risk category, confidence ring chart, top contributing factors |

Using `st.container(key=...)` rather than manually inserting `<div>` tags around each section ensures every widget placed inside a `with` block is a genuine child of that container in the DOM. This matters because Streamlit renders each call — `st.markdown`, `st.slider`, `st.columns`, and so on — as its own independent element; styling was therefore applied to the container itself, so the card background, padding, and shadow wrap all of its contents as a single visual unit rather than each element receiving its own separate box.

### Styling

The application uses a scoped, custom CSS layer (light card-based theme, Inter typeface, colour-coded risk indicators) rather than Streamlit's default appearance, so the tool reads as a clinical product rather than a generic data-science demo. Custom CSS also explicitly overrides widget label colour (`[data-testid="stWidgetLabel"] p`), since the base Streamlit theme defaults to light label text that is not legible against the white cards used here.

### Input Handling

- Numeric and categorical inputs are captured with sliders, selectboxes, number inputs, and a toggle for admission status, with dependent fields (room type, length of stay) disabled when a patient is not marked as admitted.
- The "missed previous appointments" slider is conditionally rendered: if a patient has zero previous appointments, there is no valid range to slide across, so the app displays a static value instead of constructing an invalid slider.

### Results and State Management

On clicking "Predict risk level," the form values are assembled into a patient record and passed to the prediction pipeline. The result is stored in `st.session_state["last_result"]`, so the results panel remains visible and stable while the user continues adjusting form inputs, rather than disappearing on every rerender until a new prediction is explicitly requested.

The results panel presents:
- A Plotly donut chart showing the probability distribution across Low, Medium, and High risk, with the confidence of the predicted class shown at its centre.
- The predicted risk label, colour-coded to match the chart.
- The top contributing factors behind the prediction (see Explainable AI Approach below).

---

## Prediction and Explainability Pipeline (predict.py)

### Purpose

`predict.py` is kept independent of the Streamlit interface so that feature engineering, scaling, and prediction logic can be reasoned about, tested, or reused without launching the application. It has one responsibility: given a raw patient input dictionary, reproduce the exact transformation the training data went through, then return a prediction and its explanation.

### Feature Reconstruction

`build_feature_row()` converts a raw patient dictionary into a single-row DataFrame that exactly matches the model's training-time feature columns:

- Direct numeric fields (age, blood pressure, blood sugar, cholesterol, BMI, admission and appointment history) are passed through directly.
- Categorical fields (gender, blood group, department, diagnosis, payment method, payment status, appointment status, room type) are one-hot encoded using the same column-naming convention used during training.
- Engineered categorical bands — blood pressure category, BMI category, blood sugar category, and age group — are derived using the same thresholds applied in preprocessing, so a patient's raw vitals are banded identically at inference time as they were during training.
- The health burden score is computed as the sum of previous admissions and previous appointments, matching the engineered feature used in training.
- Missed previous appointments is clipped to never exceed previous appointments, guarding against an inconsistent combination of inputs regardless of what the interface allows.

### Scaling and Prediction

The row is scaled using the exact fitted `StandardScaler` loaded from `models/scaler.pkl`, applied to the same numeric columns, in the same order, that were scaled during training. The scaled row is then passed to `model.predict_proba()`, and the class with the highest probability is returned as the prediction, alongside the full probability distribution and the model's confidence in the predicted class.

---

## Explainable AI Approach

A prediction without explanation is of limited use in a clinical context, so every prediction is paired with a SHAP-based explanation.

- **Explainer choice** — A `shap.LinearExplainer` is used, appropriate for the selected Logistic Regression model. It is built once, against a background sample of 100 rows drawn from the training data, and cached via `st.cache_resource` so it is not rebuilt on every prediction.
- **Per-class SHAP values** — For each prediction, SHAP values are computed for the specific class that was predicted, not a generic aggregate across classes.
- **Top factors** — The five features with the largest absolute SHAP contribution to the predicted class are selected and surfaced to the user.
- **Class-relative direction phrasing** — Each contributing factor is labeled as either "supports" or "argues against" the predicted class, rather than a generic "increases risk" / "decreases risk" label. This distinction was deliberate: a positive SHAP value on a "Low" risk prediction supports the Low classification — it does not mean the feature is dangerous. A generic "increases risk" label would be actively misleading for Low and Medium predictions, so the explanation logic is explicit about what each SHAP value is relative to.

This mirrors the explainability analysis carried out separately in the project's explainable AI notebook, so the reasoning surfaced inside the live application is consistent with, and traceable back to, that validated analysis.

---

## How to Run This Component

1. **Activate the project's virtual environment** from the project root:

   ```bash
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS / Linux
   ```

2. **Install dependencies** (only needed once, or after `requirements.txt` changes):

   ```bash
   pip install -r requirements.txt
   ```

3. **Confirm the required model artifacts exist** in `models/`:
   - `final_model_selection.json`
   - `model_metadata.json`
   - `scaler.pkl`
   - The selected model file (e.g. `logistic_regression.pkl`)

   These are produced by the preprocessing and model development notebooks and must be present before the application will start.

4. **Run the application** from the `app/` directory:

   ```bash
   cd app
   streamlit run app.py
   ```

5. Streamlit will start a local server and print a URL, typically:

   ```
   Local URL: http://localhost:8501
   ```

   Open this URL in a browser to use the application.

---

## Summary

This contribution covers the full path from a trained model artifact to a usable, explainable clinical tool: an interface for capturing patient data, a pipeline that faithfully reproduces the training-time feature engineering and scaling at inference time, and a SHAP-based explanation layer that surfaces per-prediction reasoning in language appropriate for a clinical, non-technical audience.