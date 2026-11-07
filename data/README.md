# Dataset

## Source

**Credit Card Fraud Detection** — anonymized European cardholder
transactions, originally released by the Machine Learning Group at
Université Libre de Bruxelles (ULB) in collaboration with Worldline.

- Original publication: Kaggle — [mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Retrieved via: [OpenML dataset 1597](https://www.openml.org/d/1597) (`creditcard`,
  file id `1673544`), which mirrors the same data in a directly
  downloadable form (no authentication required)
- Direct download used for this project:
  `https://api.openml.org/data/v1/download/1673544/creditcard.arff`
  (converted to CSV — see below)

## File

`data/raw/transactions.csv` is **not committed to this repository** (145MB,
over GitHub's comfortable size limit). Regenerate it with:

```bash
python src/download_data.py
```

## Shape

| | |
|---|---|
| Rows | 284,807 transactions |
| Columns | 31 |
| Fraud cases (`Class = 1`) | 492 (0.173%) |
| Legitimate cases (`Class = 0`) | 284,315 (99.827%) |
| Time span | ~48 hours (two consecutive days, September 2013) |

## Columns

| Column | Type | Description |
|---|---|---|
| `Time` | numeric | Seconds elapsed between each transaction and the first transaction in the dataset (**not** a wall-clock timestamp) |
| `V1`–`V28` | numeric | Principal components from a PCA transformation of the original transaction features. The original features (merchant, cardholder, location, etc.) were anonymized this way for confidentiality — their real-world meaning is not recoverable. |
| `Amount` | numeric | Transaction amount |
| `Class` | binary (0/1) | **Original target column.** Renamed to `is_fraud` during preprocessing (notebook 01 / `src/preprocessing.py`) — `0` = legitimate, `1` = fraudulent |

## Data limitations

- **Anonymized by design**: `V1`–`V28` are PCA components, not interpretable
  business features. This makes feature-level business explanations
  (e.g. "fraud is higher for X merchant category") impossible — only
  `Time` and `Amount` are human-readable.
- **`Time` is relative, not absolute**: it's seconds since the first
  transaction in the dataset, not a real timestamp. `hour_of_day` derived
  from it (Phase 6) is only meaningful modulo 24h and assumes the dataset
  starts at hour 0 — it is an approximation, not a verified clock time.
- **No customer or merchant identifiers**: features like
  "transactions per customer" (Phase 6) are **not available** in this
  dataset and are explicitly out of scope here — see the feature
  engineering notebook for what was actually derived versus what a real
  system would additionally have.
- **Extremely imbalanced**: 0.173% fraud rate. Ordinary accuracy is
  meaningless here (see Phase 9–12).
- **Narrow time window**: two days of data from one processor
  (Worldline) and one card scheme population (European cardholders) —
  findings should not be generalized to other markets, card types, or
  time periods without validation.

## Synthetic / public / anonymised?

- **Not synthetic** — these are real, historical card transactions.
- **Public** — freely available for research and educational use via
  Kaggle/OpenML.
- **Anonymised** — original features were transformed via PCA
  specifically to prevent re-identification of cardholders or merchants;
  `Time` and `Amount` are the only fields left in their original form.

This project is a portfolio simulation built on public data. It is not
connected to, and does not represent, any real financial institution's
production fraud-detection system.
