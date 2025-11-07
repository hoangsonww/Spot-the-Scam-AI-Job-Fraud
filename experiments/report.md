# Spot the Scam Report - linear_svm_C1.0

## Configuration Snapshot
- Model: **linear_svm_C1.0** (classical)
- Calibration: isotonic
- Decision threshold: 0.5452
- Gray-zone width: 0.1

## Metrics Overview
| split      |       f1 |   precision |   recall |   roc_auc |   pr_auc |     brier |
|:-----------|---------:|------------:|---------:|----------:|---------:|----------:|
| validation | 0.810606 |    0.938596 | 0.713333 |  0.988162 | 0.871819 | 0.0117084 |
| test       | 0.789272 |    0.919643 | 0.691275 |  0.981442 | 0.845352 | 0.0141198 |

## Token Signals (Top 20)
| token      |   positive_count |   negative_count |   difference |
|:-----------|-----------------:|-----------------:|-------------:|
| solutions  |               89 |             1124 |        -1035 |
| support    |               82 |             1195 |        -1113 |
| -          |               86 |             1959 |        -1873 |
| &          |              105 |             1997 |        -1892 |
| customer   |               85 |             2175 |        -2090 |
| this       |               84 |             2420 |        -2336 |
| from       |              114 |             2579 |        -2465 |
| new        |               49 |             2543 |        -2494 |
| business   |               68 |             2602 |        -2534 |
| all        |               97 |             2951 |        -2854 |
| at         |               63 |             2992 |        -2929 |
| your       |              103 |             3364 |        -3261 |
| team       |               78 |             3574 |        -3496 |
| have       |               72 |             4023 |        -3951 |
| experience |              119 |             4527 |        -4408 |
| an         |              109 |             4528 |        -4419 |
| work       |              138 |             4690 |        -4552 |
| will       |               75 |             4652 |        -4577 |
| that       |              111 |             4702 |        -4591 |
| or         |              183 |             4952 |        -4769 |

## Slice Analysis
| slice    | category               |   count |       f1 |   precision |   recall |    roc_auc |   pr_auc |       brier |
|:---------|:-----------------------|--------:|---------:|------------:|---------:|-----------:|---------:|------------:|
| function | <missing>              |     780 | 0.918033 |    1        | 0.848485 |   0.997079 | 0.948441 | 0.00647124  |
| function | Administrative         |     111 | 0.882353 |    0.9375   | 0.833333 |   0.994922 | 0.967392 | 0.0254864   |
| function | Business Development   |      34 | 1        |    1        | 1        |   1        | 1        | 6.59215e-05 |
| function | Customer Service       |     153 | 0.7      |    0.875    | 0.583333 |   0.981678 | 0.829523 | 0.0292127   |
| function | Design                 |      62 | 0        |    0        | 0        |   1        | 1        | 0.00956506  |
| function | Engineering            |     226 | 0.869565 |    1        | 0.769231 |   0.997404 | 0.977337 | 0.0166987   |
| function | Health Care Provider   |      53 | 0        |    0        | 0        | nan        | 0        | 0.0221094   |
| function | Human Resources        |      40 | 1        |    1        | 1        |   1        | 1        | 3.77718e-05 |
| function | Information Technology |     294 | 0.4      |    1        | 0.25     |   0.846552 | 0.553401 | 0.00816128  |
| function | Management             |      52 | 0        |    0        | 0        |   1        | 1        | 0.0114809   |
| function | Marketing              |     146 | 1        |    1        | 1        |   1        | 1        | 0.000179829 |
| function | Not Mentioned          |     717 | 0.666667 |    0.826087 | 0.558824 |   0.944901 | 0.672779 | 0.0238961   |
| function | Other                  |      56 | 0.666667 |    1        | 0.5      |   0.990741 | 0.75     | 0.016602    |
| function | Sales                  |     254 | 0.8      |    0.8      | 0.8      |   0.991165 | 0.858824 | 0.0112652   |

## Additional Visuals
- `experiments/figs/score_distribution_test.png`: probability density by class.
- `experiments/figs/threshold_sweep_val.png`: precision/recall/F1 trade-offs across thresholds.
- `experiments/figs/probability_vs_length.png`: regression view of text length vs fraud probability.
- `experiments/figs/latency_throughput.png`: latency vs throughput benchmark; see `experiments/tables/benchmark_summary.csv`.

## Notes
All metrics computed on the frozen test split. Gray-zone policy maps probabilities within the band to human review.