# Viva Preparation Notes

The viva is **individual** (5 marks) — each person must be able to explain
their own task in depth, AND have a working understanding of the whole
pipeline. Use this as a checklist while preparing.

## General Questions (everyone should be able to answer)

- Why was Option C (Disease Risk Classification) chosen over A or B?
- What is `disease_risk_level` and how are Low/Medium/High defined in the data dictionary?
- Walk through the pipeline end-to-end: raw data → preprocessing → EDA → models → evaluation → SHAP → prototype.
- Why macro-F1 rather than plain accuracy for model comparison? (Answer: fairness across classes if imbalanced.)
- What would you do differently with more time/data?

## Chanuu — Task 03 (Preprocessing & Feature Engineering)

- Why median imputation for numeric columns and mode for categorical?
- Why IQR-based capping instead of dropping outlier rows? (Answer: preserves sample size with only 1000 records.)
- What new features were engineered (e.g. `bmi_category`, `bp_flag`, `total_charges_calculated`) and why might they help predict risk level?
- Why drop ID-like columns before modeling? (Data leakage / no predictive value.)
- How was the train/test split stratified, and why does that matter for a multi-class target?

## Avishka — Task 04 (EDA)

- What did the class distribution chart reveal — is `disease_risk_level` imbalanced?
- Which features showed the strongest correlation with risk level, and did that match clinical intuition?
- Walk through one boxplot and one scatterplot — what pattern did it reveal?
- Were any data quality issues discovered during EDA that fed back into Task 03?

## Thrithwaka — Task 05 (Model Development)

- Why these three models (Logistic Regression, Random Forest, XGBoost)? What does each assume?
- What hyperparameters were tuned for each, and what did GridSearchCV select?
- Why `class_weight="balanced"` for Random Forest?
- What does `multi:softprob` mean for XGBoost, and why is it needed for 3 classes?

## Ramda — Task 06 (Model Evaluation)

- Explain each metric (accuracy, precision, recall, F1) in the context of THIS problem — e.g. what does a false negative mean for "High risk" patients?
- Why macro-averaging instead of weighted/micro?
- Walk through the confusion matrix for the best model — where does it struggle most?
- Why was [best model] chosen as final, and what would change that decision?

## Tharindu — Task 07 & 08 (Explainable AI + Prototype)

- Why SHAP over LIME for this project?
- Explain the difference between global (summary plot) and local (waterfall plot) explanations.
- Walk through one patient's prediction — what were the top contributing features and why?
- What are the ethical implications of using this model in real clinical decision support? (bias, over-reliance, need for clinician oversight)
- Demo the Streamlit app — what happens when you change one input value?

## Things to double check before the viva

- [ ] Everyone has actually **run** the full pipeline once on their own machine, not just read the code.
- [ ] Everyone can explain a teammate's confusion matrix / SHAP plot at a basic level, not just their own section.
- [ ] Screenshots and video demo are recorded and match the final code in the repo (no last-minute unrecorded changes).
