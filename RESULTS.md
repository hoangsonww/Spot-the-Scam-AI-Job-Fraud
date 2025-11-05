# Results

This document summarizes the results of the text classification experiment conducted using a logistic regression model with TF-IDF features. The model was trained to classify text data into binary categories.

## Performance Snapshot
| split |       f1 | precision | recall | roc_auc | pr_auc |    brier |
|:------|--------:|----------:|-------:|--------:|-------:|---------:|
| validation | 0.9048 | 0.9223 | 0.8879 | 0.9942 | 0.9335 | 0.0070 |
| test | 0.8187 | 0.9186 | 0.7383 | 0.9736 | 0.8299 | 0.0135 |

- Expected calibration error (test): `0.0085`
- Decision threshold: `0.4963`
- Gray-zone width: `0.10`

## Model Diagnostics
### Precision-Recall
![PR Curve](experiments/figs/pr_curve_test.png)

### Calibration
![Calibration Curve](experiments/figs/calibration_curve_test.png)

### Confusion Matrix
![Confusion Matrix](experiments/figs/confusion_matrix_test.png)

### Score Distribution
![Score Distribution](experiments/figs/score_distribution_test.png)

### Threshold Sweep (Validation)
![Threshold Sweep](experiments/figs/threshold_sweep_val.png)

### Probability vs. Text Length
![Probability vs Text Length](experiments/figs/probability_vs_length.png)

### Inference Benchmark
![Latency vs Throughput](experiments/figs/latency_throughput.png)

## Explainability & Insights
- Token coefficients: `experiments/tables/top_terms_positive.csv` and `.../top_terms_negative.csv`
- Token frequency deltas: `experiments/tables/token_frequency_analysis.csv`
- Slice metrics: `experiments/tables/slice_metrics.csv`
- Probability regression stats: `experiments/tables/probability_regression.csv`
- Threshold sweep data: `experiments/tables/threshold_metrics.csv`
- Latency summary: `experiments/tables/benchmark_summary.csv`

All supporting CSVs live in `experiments/tables/` for reproducible analysis.
