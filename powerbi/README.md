# Power BI — Fraud Risk Analytics (4 dashboards)

Build kit, not a `.pbix` — Power BI Desktop has no scriptable/headless
authoring API, so a binary report file can't be produced outside the
Desktop app itself. This gets you from a blank Power BI Desktop window to
the finished report quickly, with the data model, relationships, and
every measure already decided.

## 1. Get the data

**Get Data → Text/CSV**, load each of these from `../data/processed/`
(produced by `notebooks/06_model_evaluation.ipynb`):

| Load as table name | Source file |
|---|---|
| `Transactions` | `data/processed/transactions.csv` |
| `ModelResults` | `data/processed/model_results.csv` |
| `FraudPredictions` | `data/processed/fraud_predictions.csv` |
| `ExceptionTransactions` | `data/processed/exception_transactions.csv` |

Check column types in Power Query: `Transactions[amount]`,
`Transactions[log_amount]` → Decimal Number; `Transactions[is_fraud]`,
`Transactions[hour_of_day]`, `Transactions[elapsed_day]` → Whole Number;
`FraudPredictions[fraud_probability]`,
`ExceptionTransactions[fraud_probability]` → Decimal Number.

## 2. Relationships

Model view:

- `FraudPredictions[transaction_id]` → `Transactions[transaction_id]` (one-to-one)
- `ExceptionTransactions[transaction_id]` → `Transactions[transaction_id]` (one-to-one)

`ModelResults` doesn't relate to the others — it's one row per model, used
standalone on Dashboard 3 with a slicer.

## 3. Add the measures

Create a blank measures table (`Measures = ROW("_", 0)`, hide the `_`
column), then add every measure from [`measures.dax`](measures.dax) via
**New Measure**. Organize into display folders `Overview`, `Risk`,
`Model Performance`, `Operations` to match the dashboards below.

## 4. Dashboards

### Dashboard 1 — Executive Overview
Five **Card** visuals: `[Total Transactions]`, `[Fraud Transactions]`,
`[Fraud Rate %]`, `[Total Transaction Value]`, `[High Risk Transactions]`.

### Dashboard 2 — Risk Monitoring
- **Bar/column chart**: Axis `FraudPredictions[risk_level]`, Values = count of rows — "Risk distribution"
- **Histogram** (or binned bar): `FraudPredictions[fraud_probability]` — "Fraud probability" distribution
- **Card**: `[High Risk Volume]` + `[Critical Risk Volume]` — "High-risk volume"
- **Line chart**: Axis `Transactions[hour_of_day]`, Values = fraud rate (count of `is_fraud=1` / count of rows) — "Fraud trends"
- **Box plot or histogram**: `Transactions[amount]` split by `Transactions[is_fraud]` — "Transaction amounts"

### Dashboard 3 — Model Performance
Add a slicer on `ModelResults[model_name]`, then five **Card** visuals
bound to `[Precision]`, `[Recall]`, `[F1 Score]`, `[ROC AUC]`, `[PR AUC]`
— these use `SELECTEDVALUE`, so they show the slicer-selected model. Add
a **Table** visual bound to all of `ModelResults` underneath for the full
Phase 12 comparison at a glance, with `[Best Model by PR-AUC]` as a
callout card.

### Dashboard 4 — Operations Review
A **Table** bound directly to `ExceptionTransactions`: `transaction_id`,
`transaction_amount`, `fraud_probability`, `risk_level`, `prediction`,
`review_status`. Add slicers for `risk_level` and `review_status` above
it. Conditional-format `risk_level` (CRITICAL = red, HIGH = orange,
MEDIUM = amber) and `review_status` (PENDING = grey, ESCALATED = red,
REVIEWED = blue, CLEARED = green). Add cards for `[Pending Review Count]`,
`[Escalated Count]`, `[Total Exposure (Exceptions)]` above the table.

**This is the dashboard to lead with in an Operations-focused interview**
— it's the one that turns a model score into a concrete, actionable queue.

## 5. A note on the data behind this

`ExceptionTransactions[review_status]` is **simulated** for this
portfolio project (assigned via weighted random draw in
`06_model_evaluation.ipynb`) — there is no real review team behind it.
Make this explicit if presenting the dashboard: it demonstrates the
*shape* of an operational review workflow, not a live one.
