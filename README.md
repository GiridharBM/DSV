# Road-Accident Severity & Intervention Prioritizer (RAIP)

A complete, working Python/Streamlit application that demonstrates **PREDICT → EXPLAIN → PRIORITISE → VISUALISE** on the official UK Department for Transport **STATS19** road collision dataset.

> **Geographic scope:** Great Britain (England, Scotland, Wales) — **not India**. STATS19 covers police-reported personal-injury collisions on public roads. Results must not be presented as Indian road-safety statistics.

---

## 1. Quick start

```bash
pip install -r requirements.txt

# 1) Download the official 2021–2025 collision CSVs from data.dft.gov.uk (~97 MB)
python -m src.data.download

# 2) Full pipeline: clean -> engineer -> train -> evaluate -> SHAP -> IPS -> reports
python scripts/run_pipeline.py

# 3) Launch the dashboard
streamlit run app.py
```

If the download fails, the app still starts: use the **manual CSV upload** in the sidebar (a STATS19-style collision CSV), or it falls back to a clearly labelled **synthetic DEMO** dataset. The demo never produces fake accuracy/SHAP values — prediction and SHAP pages show an explicit unavailable state if no model artifacts exist.

Rebuild only the ranking/report artifacts (no retraining) with `python scripts/run_reports.py`.
Run tests with `python -m pytest tests -q`.

---

## 2. Data

- **Source:** [DfT Road safety open data](https://www.gov.uk/government/statistical-data-sets/road-safety-open-data), collision-level CSVs (currently 2021–2025; ~514k records after cleaning).
- Downloaded files are preserved unmodified in `data/raw/collision_YYYY.csv`.
- Cleaned data: `data/processed/collisions_clean.csv`; provenance: `data/processed/provenance.json`.
- **Severity mapping (official):** `collision_severity` 1 = Fatal, 2 = Serious, 3 = Slight.
- `-1` in coded fields means "data missing or out of range" and is mapped to `Data missing`/`Unknown`.
- Deduplication uses `collision_index`; invalid dates/severities are removed; numeric gaps use the median.
- No casualty, attendance or other post-collision fields are used as model features. `number_of_casualties` and `did_police_officer_attend_scene_of_accident` are retained **only** in the processed file for auditing/exploration, never in `FEATURE_COLUMNS`.

### Feature contract (`src/config.py`)
Numerics: vehicles involved, speed limit, hour, month, weekend flag, night flag.
Categoricals: day, lighting, weather, road surface, road type, junction detail/control, site conditions, carriageway hazards, urban/rural, road class, time-of-day band.

---

## 3. Machine learning

- **Chronological evaluation:** train on 2021–2024, test on 2025 (no random split across years). A seeded uniform cap (`RAIP_MAX_MODEL_ROWS`, default 150,000) limits training time; the full holdout year is evaluated.
- **Models compared:** Logistic Regression (balanced), Random Forest (balanced subsample), XGBoost (balanced sample weights, hist tree method).
- **Selection rule:** `macro_F1 + 0.05 × (Fatal recall + Serious recall)` — explicit emphasis on the rare severe classes. Accuracy is reported but never used alone.
- Preprocessing (median/mode imputation, one-hot encoding, scaling) is fitted **inside** each pipeline, so there is no leakage from test into train.
- Artifacts: `models/best_model.joblib`, per-model joblibs, `models/model_meta.json`. The prediction page uses the actual saved pipeline.

Evaluation outputs in `reports/`: `model_comparison.csv`, `model_classwise.csv`, `model_confusion.csv`, `evaluation_summary.json`, and `summary_report.pdf`.

> Real baseline result on this dataset: best model Random Forest, macro-F1 ≈ 0.36, accuracy ≈ 0.52, Fatal recall ≈ 0.43 on the 2025 holdout. Severe-class prediction is intrinsically hard with these variables; these are genuine measured values, not placeholders.

---

## 4. SHAP explainability

- `TreeExplainer` for tree models (or `LinearExplainer` for logistic regression) is run on a seeded sample of the actual holdout set.
- Outputs: `reports/shap_global_importance.csv`, `reports/shap_category_importance.csv`, `models/shap_values.npy`, `models/shap_meta.json`.
- The dashboard shows global importance, a beeswarm, category-level importance and individual contribution plots — all derived from the trained model. If SHAP is unavailable, the UI says so rather than inventing values.

---

## 5. Intervention Priority Engine (IPS)

For each condition (feature = category pair, e.g. `Weather = Fog or mist`, minimum 150 collisions):

```
IPS_i = w_f · F_i + w_s · S_i + w_e · E_i
```

- **F** — collisions in the condition ÷ the largest condition count (with denominators shown).
- **S** — mean analytic severity impact per collision (Fatal = 5, Serious = 3, Slight = 1) ÷ the highest condition mean. These weights are an **analytic convention**, not official DfT monetary costs.
- **E** — mean absolute SHAP contribution of the matching encoded category (or its parent feature as fallback) ÷ the largest value.

Weights are non-negative and normalised to sum to 1; sliders update the ranking live. The centre provides component breakdowns, frequency-only / severity-only / SHAP-only ranks, Spearman stability, top-10 overlap and a weight-sensitivity sweep. Output: `reports/ips_ranking.csv`, `reports/ranking_stability.csv`, `reports/weight_sensitivity.csv`.

**Interpretation limits:** associations are not causation; traffic exposure, reporting practice and confounding are not modelled. The score does **not** claim that an intervention will work or that collisions will be reduced.

---

## 6. Dashboard pages

| Page | Contents |
|---|---|
| Executive Overview | KPIs, severity mix, annual trends, weather/road/light/time analysis, auto insights |
| Accident Explorer | Date, severity, weather, road, lighting, surface filters; search; CSV export |
| Severity Prediction | Form over the true feature contract, probabilities, SHAP explanation |
| Explainable AI | Global importance, category importance, beeswarm, individual explanations |
| Intervention Priority | IPS ranking, weight sliders, components, comparisons, sensitivity (centrepiece) |
| Model Performance | Comparison, macro-F1, class-wise recall, confusion matrix, classification report |
| Geographic Intelligence | Real lat/lon map, filters, severity by area, grid hotspots, honest unavailable states |

Design: navy background, teal/cyan accents, severity red/orange/blue indicators, rounded KPI cards, Plotly charts, responsive Streamlit layout.

---

## 7. Reliability

- Reproducible seeds (`RANDOM_SEED = 42`), cached data/model loading, structured logging, explicit validation and error handling.
- Automatic insights are computed from data with denominators; nothing is hardcoded.
- Synthetic demo data is labelled everywhere and never used for fake model metrics.
- Empty-data and missing-artifact states are handled on every page.

---

## 8. Repository layout

```
app.py                     Streamlit entry point
dashboard/pages/           Seven analysis pages
dashboard/components/      Reusable UI helpers
dashboard/theme.py         Dark visual theme
src/data/                  Download, loader, demo generator
src/preprocessing/         Validation, cleaning, encoder
src/features/              Leakage-free feature engineering
src/models/                Training and persistence
src/evaluation/            Metrics and report writers
src/explainability/        SHAP
src/prioritization/        IPS, stability, sensitivity
src/utils/                 Logging, insights
scripts/                   Pipeline and report runners
tests/                     Automated tests
data/raw|processed/        Source and clean data
models/                    Trained artifacts
reports/                   Evaluation, SHAP and IPS outputs
```
