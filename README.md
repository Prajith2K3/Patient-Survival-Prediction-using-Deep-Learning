[README.md](https://github.com/user-attachments/files/30933311/README.md)
# Patient Survival Prediction & Clinical Risk Analytics

Dense neural network predicting in-hospital death (`hospital_death`) from ICU admission data, using the architecture and training configuration previously specified for this project: `Dense(12,relu) → Dense(8,relu) → Dense(1,sigmoid)`, Adam(lr=0.02), binary cross-entropy, 20 epochs, batch size 10.

> **Data note:** The real GOSSIS-based ICU dataset (~91,713 rows × 186 columns, as previously described in project context) could not be retrieved in this environment — no network access to Kaggle/data-source hosts is available here. A **documented synthetic ICU dataset (8,000 patients, 22 features)** was generated instead, using the clinically-named feature subset previously mentioned (APACHE probabilities, GCS components, vitals, labs, comorbidities), with hand-specified relationships to `hospital_death`. This lets the exact architecture, feature-selection method, training configuration, and SHAP explainability step run end-to-end and produce real, verifiable outputs — **but the resulting metrics below are not the same as, and should not be presented as, any previously reported figure from the real GOSSIS dataset.** Verifying against the original data/notebook is still required before quoting any number here as a historical result.
>
> **This is an ML prediction prototype, not a clinically validated or deployed healthcare system.**

## Pipeline
1. **Cleaning** — median imputation on lab columns with injected missingness
2. **Feature selection** — `mutual_info_classif` scores all 22 features, `SelectKBest(k=12)` selects the top 12
3. **Scaling** — `StandardScaler` on selected features
4. **Model** — `Dense(12,relu) → Dense(8,relu) → Dense(1,sigmoid)`, Adam(lr=0.02), binary cross-entropy, 20 epochs, batch_size=10, validation_split=0.2
5. **Evaluation** — accuracy, precision, recall, F1, ROC-AUC, confusion matrix, training curves
6. **Explainability** — SHAP KernelExplainer summary plot over the trained network

## Selected features (top 12 by mutual information)
`age`, `heart_rate_apache`, `gcs_motor_apache`, `gcs_eyes_apache`, `gcs_verbal_apache`, `apache_3j_diagnosis`, `ventilated_apache`, `diabetes_mellitus`, `creatinine_apache`, `sodium_apache`, `apache_4a_icu_death_prob`, `apache_4a_hospital_death_prob`

## Results (this run, on synthetic data)

| Metric | Test Set |
|---|---|
| Accuracy | 87.6% |
| Precision | 75.5% |
| Recall | 67.1% |
| F1 | 71.1% |
| ROC-AUC | 0.921 |

Training accuracy 88.0% / validation accuracy 86.5% — no major overfitting across 20 epochs (see training curves plot). Full breakdown in `classification_report.txt`.

## Files in this package
```
patient_survival_prediction/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── data/
│   ├── patient_survival_dataset.csv
├── docs/
│   ├── classification_report.txt
├── python/
│   ├── Patient_Survival_Prediction.ipynb
│   ├── patient_survival_prediction.py
│   ├── streamlit_app.py
└── images/
    ├── 01_outcome_distribution.png
    ├── 02_eda_relationships.png
    ├── 03_feature_selection_mi_scores.png
    ├── 04_training_curves.png
    ├── 05_confusion_matrix.png
    ├── 06_roc_curve.png
    └── 07_shap_summary.png
```

## Suggested LinkedIn caption

> 🏥 **Patient Survival Prediction — Deep Learning on ICU Data**
>
> Built a dense neural network to predict in-hospital mortality risk from ICU admission features (APACHE scores, GCS, vitals, labs).
>
> 🔧 What I did:
> • Selected the top 12 predictive features with mutual information + SelectKBest
> • Trained a 3-layer neural network (Dense 12→8→1, Adam, 20 epochs) — 87.6% accuracy, 0.921 ROC-AUC
> • Added SHAP explainability to interpret individual predictions
> • Sketched a Streamlit UI for interactive risk scoring
>
> 🛠️ Tools: Python, TensorFlow/Keras, Scikit-Learn, SHAP, Streamlit
>
> ⚕️ Academic/portfolio project — not a clinically validated tool.
>
> #DataScience #DeepLearning #Healthcare #TensorFlow #Portfolio

## Limitations & ethical notes
- Synthetic data with hand-specified relationships — not validated against real ICU records or real GOSSIS data
- Not deployed or clinically validated; no monitoring, drift-detection, or governance pipeline in place
- Real deployment would require IRB/clinical oversight, external validation, subgroup bias auditing, and a maintained model-monitoring pipeline
