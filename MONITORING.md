# Drift monitoring analysis

## Measured setup

The reference dataset contains the **27,126 training rows**, while simulated
production contains **3,000 rows sampled from the disjoint holdout pool** with
seed 123. Targets are excluded. Every one of the 15 configured model input
features is checked. Reference preprocessing is not refitted on production.
Monitoring compares raw distributions and preserves missing values; it does
not run the model's imputer before measuring drift.

Evidently 0.6.7 uses normalized Wasserstein distance for numeric features and
Jensen–Shannon distance for categoricals, with a per-feature distance threshold
of **0.10**. Scores below are distances, not p-values. The global alert fires
when more than **20%** of features drift.

## Which features drifted, and why?

The baseline batch had **0 of 15** drifting features and exited **0**.
The deliberately shifted batch had **4 of 15 (26.67%)** drifting features
and exited **1**, correctly exceeding the global alert threshold.

| Feature | Simulated change | Drift score | Detected? |
| --- | --- | ---: | --- |
| age | Add 20 years, cap at 100 | 1.9009 | Yes |
| balance | Add 10,000 to account balance | 3.2108 | Yes |
| campaign | Add 8 contacts | 2.5893 | Yes |
| contact | Set 90% of the sampled rows to telephone | 0.6451 | Yes |

These changes represent an older customer population, higher account
balances, more intensive outreach, and a different contact-channel mix.
They are deliberate simulations rather than evidence of changes at a real bank.
The remaining 11 features stayed below the configured distance threshold.

The HTML report provides per-feature distributions; `reports/drift_summary.json`
records feature flags, scores, methods, and missingness shares. The baseline
summary is `reports/drift_baseline_summary.json`.

## Would this likely affect model performance?

It could. Age, balance, campaign frequency, and contact channel are inputs to
the trained model, so these changes may move customers into less familiar
regions or change the mix of cases for which its predictions were validated.
For example, a heavier telephone mix may change which customers respond to a
campaign. However, input drift alone does not establish a decline in ROC-AUC,
precision, recall, or calibration. Some shifts can leave performance stable.

The selected model scored **0.7846 ROC-AUC** and **0.4342 average precision** on
the original reserved holdout. These results do **not** describe the artificially
shifted population. Editing features while preserving historical labels does
not create valid new labels; therefore this report does not claim a measured
performance decline after the simulated edits. New observed outcomes are
needed to establish that.

## Recommended action

**Investigate the alert first.** Check whether upstream parsing, currency
units, defaults, or campaign ETL changed. Examine the unexpected channel mix
and numeric distributions with the data owner. Compare against expected
campaign seasonality and approved targeting changes.

If the input data is wrong, correct the data issue before considering model
retraining. If the population change is real, gather labeled recent outcomes,
evaluate performance and subgroup behavior, and compare a candidate retrained
model against the current model on a fresh temporal holdout. Retrain and
release only if this evidence supports it and configured gates pass. If
performance and business outcomes remain acceptable, keep the model and
continue monitoring the affected features more closely.

The stable simulation supports continued monitoring for that batch. In a
production system, alerts would be scheduled, routed to an owner, and paired
with missingness/schema checks and labeled performance dashboards. This
project generates local reports and exit codes; it does not send alerts or
automatically retrain a model.
