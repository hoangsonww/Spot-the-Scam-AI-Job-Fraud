# Spot the Scam Report — logistic_regression_C10.0

## Configuration Snapshot
- Model: **logistic_regression_C10.0** (classical)
- Calibration: isotonic
- Decision threshold: 0.7189
- Gray-zone width: 0.1

## Metrics Overview
| split      |       f1 |   precision |   recall |   roc_auc |   pr_auc |     brier |
|:-----------|---------:|------------:|---------:|----------:|---------:|----------:|
| validation | 0.798479 |    0.921053 | 0.704698 |  0.982626 | 0.853099 | 0.0126915 |
| test       | 0.758893 |    0.932039 | 0.64     |  0.979048 | 0.862215 | 0.0131639 |

## Token Signals (Top 20)
| token      |   positive_count |   negative_count |   difference |
|:-----------|-----------------:|-----------------:|-------------:|
| *          |               84 |               62 |           22 |
| -          |              118 |             2021 |        -1903 |
| &          |               84 |             2057 |        -1973 |
| new        |               69 |             2535 |        -2466 |
| by         |               69 |             2592 |        -2523 |
| from       |               93 |             2665 |        -2572 |
| business   |               35 |             2667 |        -2632 |
| all        |               89 |             2967 |        -2878 |
| at         |               61 |             2941 |        -2880 |
| your       |              102 |             3380 |        -3278 |
| team       |               61 |             3713 |        -3652 |
| have       |               56 |             4037 |        -3981 |
| experience |              149 |             4609 |        -4460 |
| will       |               92 |             4691 |        -4599 |
| an         |               86 |             4695 |        -4609 |
| or         |              169 |             4846 |        -4677 |
| work       |              128 |             4816 |        -4688 |
| that       |               70 |             4766 |        -4696 |
| on         |              138 |             5499 |        -5361 |
| be         |              147 |             5663 |        -5516 |

## Slice Analysis
| slice    | category               |   count |       f1 |   precision |   recall |    roc_auc |   pr_auc |       brier |
|:---------|:-----------------------|--------:|---------:|------------:|---------:|-----------:|---------:|------------:|
| function | <missing>              |     771 | 0.766667 |    0.958333 | 0.638889 |   0.977343 | 0.885849 | 0.0117501   |
| function | Administrative         |     100 | 0.933333 |    0.954545 | 0.913043 |   0.997177 | 0.98444  | 0.0158018   |
| function | Business Development   |      36 | 0        |    0        | 0        | nan        | 0        | 7.15781e-05 |
| function | Customer Service       |     147 | 0.777778 |    1        | 0.636364 |   0.989305 | 0.882634 | 0.0221136   |
| function | Design                 |      67 | 0        |    0        | 0        | nan        | 0        | 6.14951e-05 |
| function | Engineering            |     242 | 0.9      |    1        | 0.818182 |   0.99845  | 0.979974 | 0.011542    |
| function | Finance                |      36 | 0        |    0        | 0        |   1        | 1        | 0.0410428   |
| function | Health Care Provider   |      53 | 0        |    0        | 0        | nan        | 0        | 0.0145654   |
| function | Human Resources        |      33 | 0        |    0        | 0        |   1        | 1        | 0.0134849   |
| function | Information Technology |     290 | 0.5      |    1        | 0.333333 |   0.813589 | 0.670127 | 0.0056724   |
| function | Management             |      37 | 0        |    0        | 0        |   1        | 1        | 0.016479    |
| function | Marketing              |     145 | 0        |    0        | 0        |   0.961806 | 0.142857 | 0.00667548  |
| function | Not Mentioned          |     758 | 0.641509 |    0.894737 | 0.5      |   0.947473 | 0.705619 | 0.0208867   |
| function | Other                  |      62 | 0.888889 |    1        | 0.8      |   0.994737 | 0.942857 | 0.016254    |
| function | Project Management     |      36 | 0        |    0        | 0        |   1        | 1        | 0.0253282   |
| function | Sales                  |     229 | 0.571429 |    0.666667 | 0.5      |   0.991667 | 0.764423 | 0.00733109  |

## Additional Visuals
- `experiments/figs/score_distribution_test.png`: probability density by class.
- `experiments/figs/threshold_sweep_val.png`: precision/recall/F1 trade-offs across thresholds.
- `experiments/figs/probability_vs_length.png`: regression view of text length vs fraud probability.
- `experiments/figs/latency_throughput.png`: latency vs throughput benchmark; see `experiments/tables/benchmark_summary.csv`.

## Notes
All metrics computed on the frozen test split. Gray-zone policy maps probabilities within the band to human review.