# Fintech Fraud Risk Analytics

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?logo=scikitlearn&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-analytics-4169E1?logo=postgresql&logoColor=white)
![Power BI](https://img.shields.io/badge/Power%20BI-dashboard-F2C811?logo=powerbi&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

An end-to-end fraud risk analytics pipeline — from a public transaction
dataset through validation, EDA, statistical testing, modelling, and
threshold-tuned risk scoring, to an operational review workflow, SQL
analysis, and an interactive dashboard.

> **Portfolio simulation.** This project uses a real, public, anonymized
> dataset, but the operational workflow (review statuses, case
> management) is simulated for demonstration. It does not represent any
> real financial institution's production fraud-detection system.

---

## Table of contents

1. [Business problem](#1-business-problem)
2. [Objective](#2-objective)
3. [Dataset](#3-dataset)
4. [Data quality](#4-data-quality)
5. [EDA](#5-eda)
6. [Feature engineering](#6-feature-engineering)
7. [Modelling](#7-modelling)
8. [Model evaluation](#8-model-evaluation)
9. [Threshold selection](#9-threshold-selection)
10. [Risk scoring](#10-risk-scoring)
11. [Operational workflow](#11-operational-workflow)
12. [Dashboard](#12-dashboard)
13. [Key findings](#13-key-findings)
14. [Business recommendations](#14-business-recommendations)
15. [Limitations](#15-limitations)
16. [Project structure & running it](#project-structure--running-it)

---

## 1. Business problem

A card issuer processes far more legitimate transactions than fraudulent
ones — in this dataset, roughly **596 legitimate transactions for every
1 fraud case**. Reviewing every transaction manually is infeasible;
ignoring the problem is costly. The business needs a way to **rank
transactions by fraud risk** so a small, high-value review queue catches
most fraud without burying Operations in false alarms.

## 2. Objective

Build a model that outputs a fraud *probability* (not just a yes/no),
convert that into an operational risk tier, and route transactions
accordingly — while being explicit about **what the model can and can't
know** (Phase 7's leakage discipline) and **why each threshold/tier
choice was made** (Phases 14-15), not just reporting a headline accuracy
number.

## 3. Dataset

**Credit Card Fraud Detection** (ULB/Worldline, via
[OpenML dataset 1597](https://www.openml.org/d/1597), mirroring
[Kaggle's mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)) —
284,807 real, anonymized European card transactions over ~48 hours,
492 (0.173%) labelled fraud. `V1`-`V28` are PCA components of the
original (private) transaction features; `Time` and `Amount` are the
only human-readable fields. Full documentation, including exactly what
this dataset does and doesn't support, is in
[`data/README.md`](data/README.md).

## 4. Data quality

([`notebooks/02_data_quality.ipynb`](notebooks/02_data_quality.ipynb))

| Check | Finding | Action |
|---|---|---|
| Missing values | 0 across all 31 columns | none needed |
| Duplicate rows | 1,081 (19 labelled fraud) | dropped — 30 continuous fields matching by chance is implausible; treated as duplicate records, and a cross-split leakage risk if kept |
| Data types | already numeric | renamed `Class` → `is_fraud` |
| Invalid values | none (no negative amounts/time, no out-of-range labels) | none needed |
| Amount outliers (IQR) | **not** associated with a higher fraud rate | **kept** — investigated, not removed |

Clean dataset: 283,726 rows.

## 5. EDA

([`notebooks/03_eda.ipynb`](notebooks/03_eda.ipynb))

- Fraud rate by amount is **U-shaped**, not monotonic: highest in the
  smallest bucket (0-1, 0.51%) — consistent with card-testing behaviour —
  dipping through 10-100, then rising again at 500-1000 (0.40%).
- Fraud rate is markedly elevated in the **2am-5am** window (up to 1.45%
  in a single hour vs. a ~0.1-0.3% baseline).
- `V14`, `V4`, `V12`, `V11`, `V10` separate the classes most strongly
  among the anonymized components.
- `V1`-`V28` are mutually orthogonal by construction (PCA) — the usual
  multicollinearity concern that motivates dropping correlated features
  mostly doesn't apply here.

**Statistical tests** (both pre-registered to answer a specific
question, not run for decoration):

| Question | Test | Statistic | p-value | Conclusion |
|---|---|---|---|---|
| Does fraud have a different amount distribution than legitimate transactions? | Mann-Whitney U (non-parametric — `Amount` is heavily right-skewed) | U = 59,517,129.5 | 2.686e-05 | Reject H0. Real difference, but modest effect size (rank-biserial r = 0.11) — amount alone is informative, not a clean separator |
| Is fraud rate independent of time-of-day (night vs. day)? | Chi-square test of independence | χ² = 322.64 | 3.85e-72 | Reject H0. The night-time elevation is statistically real on this sample |

## 6. Feature engineering

([`notebooks/04_feature_engineering.ipynb`](notebooks/04_feature_engineering.ipynb),
[`src/features.py`](src/features.py))

| Feature | Source | Rationale |
|---|---|---|
| `log_amount` | `Amount` | Variance-stabilizing transform of a right-skewed field |
| `hour_of_day`, `elapsed_day` | `Time` | Captures the time-of-day pattern found in EDA |
| `transactions_per_hour_bucket` | `Time` | Volume/context signal — is this an unusually busy hour |

**Deliberately not created**: `transactions_per_customer` /
`amount_vs_customer_average` (no customer identifier exists in this
dataset — fabricating one would invent information, not engineer a
feature) and `day_of_week` (`Time` is relative-seconds, not a real
calendar date).

## 7. Modelling

([`notebooks/05_model_training.ipynb`](notebooks/05_model_training.ipynb))

**Data leakage check (Phase 7)** — every feature was tested against
"could this be known before a fraud decision is made, without seeing a
future investigation outcome?" No leakage-risk columns exist in the raw
data itself; the two real risks in this *pipeline* were (1) duplicate
rows spanning train/test — fixed by dropping duplicates before splitting
— and (2) oversampling before the split — avoided procedurally by
applying SMOTE only to the training fold, after splitting.

**Split**: stratified 60/20/20 train/validation/test (fraud rate held
constant across all three).

**Models**: Logistic Regression (baseline), Random Forest, Gradient
Boosting — three, not fifteen; the goal was a clean comparison, not an
algorithm survey.

**Class imbalance**: `class_weight='balanced'` (Logistic Regression,
Random Forest) / balanced sample weights (Gradient Boosting), validated
against an unweighted baseline and against SMOTE:

| Approach | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| LR — unweighted | 0.860 | 0.521 | 0.649 | 0.695 |
| LR — class_weight=balanced | 0.054 | 0.851 | 0.102 | 0.706 |
| LR — SMOTE | 0.052 | 0.851 | 0.099 | 0.704 |

Unweighted recall (0.52) is dramatically lower than either balancing
approach (0.85) — confirming imbalance handling matters here, not just
in theory. `class_weight='balanced'` was carried forward (applies
identically across all three model families; no training-data copy
needed).

## 8. Model evaluation

([`notebooks/06_model_evaluation.ipynb`](notebooks/06_model_evaluation.ipynb))

Test set (56,746 transactions, 95 fraud), default 0.5 threshold shown
for comparability across models:

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.0574 | 0.9053 | 0.108 | 0.9627 | 0.7328 |
| **Random Forest** | **0.8765** | **0.7474** | **0.8068** | **0.9852** | **0.8117** |
| Gradient Boosting | 0.2314 | 0.8842 | 0.3668 | 0.9773 | 0.7594 |

**Random Forest selected — by PR-AUC, not accuracy.** With 99.83% of
transactions legitimate, accuracy is meaningless (predicting "legitimate"
for everything scores ~99.83%). ROC-AUC stays flattered by the huge
true-negative population even when positive-class precision is weak.
**PR-AUC** is computed entirely on precision/recall for the fraud class —
the tradeoff Operations actually lives with — so it drove model
selection here.

## 9. Threshold selection

The model outputs a probability; 0.5 is an arbitrary default, not a
property of the business problem. A full precision/recall/FP/FN sweep
(on validation, so the choice isn't fitted to the test set) is in
notebook 06. **This project's stated objective**: minimize missed fraud
(a realized loss) over minimizing false positives (a review-capacity
cost) — i.e. bias toward recall.

**Chosen threshold: 0.30** (this project's own choice for its own stated
objective — not a claimed industry standard). At this threshold, on the
held-out test set:

```
                 Predicted
              Legit    Fraud
Actual Legit  56,622       29
Actual Fraud      18       77
```

Precision 0.726, Recall 0.811, F1 0.766. **18 fraud cases missed** (the
costliest error — undetected loss) against **29 legitimate transactions
flagged** (a review-capacity cost, not a financial loss) — a deliberate
trade favoring catching more fraud.

## 10. Risk scoring

Fraud probability converted into this project's own configurable risk
tiers (`src/evaluation.RISK_BANDS` — not a universal banking standard):

| Tier | Probability | Test-set count |
|---|---|---|
| LOW | 0-20% | 56,573 |
| MEDIUM | 20-50% | 92 |
| HIGH | 50-80% | 13 |
| CRITICAL | 80-100% | 68 |

`data/processed/fraud_predictions.csv`: one row per test transaction —
`transaction_id`, `fraud_probability`, `risk_level`, `model_prediction`.

## 11. Operational workflow

```
                 Transaction
                      |
                Risk Model
                      |
                Risk Score
                      |
          +-----------+-----------+
          |                       |
       LOW/MEDIUM             HIGH/CRITICAL
          |                       |
     Normal processing       Manual review
                                  |
                             Investigation
                                  |
                              Escalation
```

81 test-set transactions land in HIGH/CRITICAL and form the exception
queue (`data/processed/exception_transactions.csv`,
`review_status` ∈ `PENDING`/`REVIEWED`/`ESCALATED`/`CLEARED`). **This
queue's review statuses are simulated** for this portfolio project — see
[Limitations](#15-limitations).

## 12. Dashboard

[`dashboard/fraud_risk_console.html`](dashboard/fraud_risk_console.html) —
self-contained, opens directly in any browser, no server required, built
from the real pipeline output:

1. **Executive Overview** — total transactions, fraud transactions,
   fraud rate, total transaction value, high-risk transactions
2. **Risk Monitoring** — fraud probability distribution, fraud rate by
   hour, transaction amount by class
3. **Model Performance** — the Phase 8 comparison table, PR-AUC by
   model, confusion matrix
4. **Operations Review** — the filterable/sortable exception queue

A native Power BI build kit (data model, relationships, DAX measures,
page-by-page layout) is in [`powerbi/README.md`](powerbi/README.md) and
[`powerbi/measures.dax`](powerbi/measures.dax).

## 13. Key findings

- **Fraud clusters at the amount extremes, not uniformly across value** —
  a U-shaped pattern (small card-testing charges and larger high-value
  fraud), with ordinary mid-range amounts safest. A single "flag anything
  over £X" rule would miss the smaller cluster entirely.
- **Fraud rate is ~6x higher in the 2am-5am window**, and this is
  statistically confirmed (χ² test), not an artifact of low volume.
- **Threshold choice moves the outcome more than model choice alone**:
  Logistic Regression's recall swings from 0.52 to 0.85 purely from
  imbalance handling, before any threshold tuning is even applied.
- **The chosen model (Random Forest, threshold 0.30) catches 77 of 95
  test-set fraud cases (81% recall)** while flagging only 29 legitimate
  transactions out of 56,651 (0.05% false-positive rate) — a queue small
  enough to be reviewable.

## 14. Business recommendations

**Finding**: High-risk transactions are disproportionately associated
with very small amounts (card-testing pattern) and the overnight window,
not spread evenly across the transaction population.

**Impact**: A flat, amount-only or time-blind review rule would both
over-flag ordinary transactions and under-flag the specific patterns
that actually carry risk here.

**Recommendation**: Route HIGH/CRITICAL-risk transactions (per the
model's probability, not a raw amount or time rule) to manual review,
let LOW/MEDIUM proceed through normal processing, and consider
concentrating review-team attention or automated secondary checks during
the overnight window where fraud rate is highest per transaction
reviewed.

**Limitation**: see below — this is a public, anonymized dataset from
one processor over two days; it does not represent a production fraud
environment.

## 15. Limitations

- The dataset is public and anonymized — `V1`-`V28` cannot be mapped
  back to real business features, so "why" a specific transaction is
  risky can only be answered in terms of anonymized components, not
  customer/merchant behaviour.
- Two days of data from one processor/population — findings should not
  be generalized to other markets, card types, or time periods without
  re-validation.
- No customer or merchant identifier exists, so no per-customer
  behavioural features (e.g. spending-pattern deviation) could be built
  — a real system would very likely have these and should use them.
- `hour_of_day` is derived from a relative time counter, not a verified
  clock — read as "a real pattern in this sample," not a claim about
  actual clock-hours.
- The operational review workflow (`review_status`) is **simulated**,
  not backed by a real review team or case-management system.
- This is a portfolio project on public data, not a production
  financial institution's fraud-detection system.

---

## Project structure & running it

```
fintech-fraud-risk-analytics/
├── data/
│   ├── raw/                  # transactions.csv (not committed — see below)
│   ├── processed/            # cleaned/feature/model outputs
│   └── README.md             # full dataset documentation
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_quality.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training.ipynb
│   └── 06_model_evaluation.ipynb
├── src/
│   ├── download_data.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── models.py
│   └── evaluation.py
├── sql/
│   ├── 01_create_tables.sql
│   └── risk_analysis.sql
├── dashboard/
│   └── fraud_risk_console.html
├── powerbi/
│   ├── README.md
│   └── measures.dax
└── requirements.txt
```

```bash
git clone https://github.com/Kourpreetika/fintech-fraud-risk-analytics.git
cd fintech-fraud-risk-analytics
pip install -r requirements.txt

python src/download_data.py     # -> data/raw/transactions.csv (145MB, not committed)

jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_understanding.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_data_quality.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_feature_engineering.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/05_model_training.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/06_model_evaluation.ipynb
```

(Or just open each notebook in Jupyter and run all cells — they're meant
to be read, not just executed.)

### Loading into PostgreSQL

```bash
psql -d your_database -f sql/01_create_tables.sql
cd sql && psql -d your_database
\copy transactions FROM '../data/processed/transactions.csv' WITH (FORMAT csv, HEADER true)
\copy model_results FROM '../data/processed/model_results.csv' WITH (FORMAT csv, HEADER true)
\copy fraud_predictions FROM '../data/processed/fraud_predictions.csv' WITH (FORMAT csv, HEADER true)
\copy exception_transactions FROM '../data/processed/exception_transactions.csv' WITH (FORMAT csv, HEADER true)
\i risk_analysis.sql
```

## Technology stack

| Layer | Tools |
|---|---|
| Data, validation, EDA, stats, modelling | Python (pandas, NumPy, scikit-learn, SciPy, imbalanced-learn) |
| Analytics | SQL (PostgreSQL) |
| Dashboard | HTML/JS (shipped) + Power BI (build kit) |
| Version control | Git / GitHub |

## License

MIT — see [LICENSE](LICENSE).

Based on [manichitumala/fintech-fraud-risk-analytics](https://github.com/manichitumala/fintech-fraud-risk-analytics).
