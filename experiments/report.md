# Spot the Scam Report - linear_svm_C1.0

## Configuration Snapshot
- Model: **linear_svm_C1.0** (classical)
- Calibration: isotonic
- Decision threshold: 0.4680
- Gray-zone width: 0.1

## Metrics Overview
| split      |       f1 |   precision |   recall |   roc_auc |   pr_auc |     brier |
|:-----------|---------:|------------:|---------:|----------:|---------:|----------:|
| validation | 0.813688 |    0.946903 | 0.713333 |  0.988126 | 0.875413 | 0.0114549 |
| test       | 0.796935 |    0.928571 | 0.697987 |  0.981723 | 0.850056 | 0.0139119 |

## Token Signals (Top 20)
| token      |   positive_count |   negative_count |   difference |
|:-----------|-----------------:|-----------------:|-------------:|
| solutions  |               89 |             1124 |        -1035 |
| support    |               82 |             1195 |        -1113 |
| -          |               85 |             1960 |        -1875 |
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
| function | <missing>              |     780 | 0.918033 |    1        | 0.848485 |   0.997059 | 0.95135  | 0.00660287  |
| function | Administrative         |     111 | 0.882353 |    0.9375   | 0.833333 |   0.995221 | 0.969059 | 0.0232007   |
| function | Business Development   |      34 | 1        |    1        | 1        |   1        | 1        | 6.92839e-05 |
| function | Customer Service       |     153 | 0.8      |    1        | 0.666667 |   0.983747 | 0.865238 | 0.0261522   |
| function | Design                 |      62 | 0        |    0        | 0        |   1        | 1        | 0.00969083  |
| function | Engineering            |     226 | 0.869565 |    1        | 0.769231 |   0.997308 | 0.975917 | 0.0163895   |
| function | Health Care Provider   |      53 | 0        |    0        | 0        | nan        | 0        | 0.0229567   |
| function | Human Resources        |      40 | 1        |    1        | 1        |   1        | 1        | 4.26555e-05 |
| function | Information Technology |     294 | 0.4      |    1        | 0.25     |   0.843534 | 0.547519 | 0.00871674  |
| function | Management             |      52 | 0        |    0        | 0        |   1        | 1        | 0.0116369   |
| function | Marketing              |     146 | 1        |    1        | 1        |   1        | 1        | 0.000124784 |
| function | Not Mentioned          |     717 | 0.666667 |    0.826087 | 0.558824 |   0.947313 | 0.688025 | 0.0230377   |
| function | Other                  |      56 | 0.666667 |    1        | 0.5      |   0.990741 | 0.75     | 0.0165515   |
| function | Sales                  |     254 | 0.8      |    0.8      | 0.8      |   0.98996  | 0.85     | 0.0113518   |

## Additional Visuals
- `experiments/figs/score_distribution_test.png`: probability density by class.
- `experiments/figs/threshold_sweep_val.png`: precision/recall/F1 trade-offs across thresholds.
- `experiments/figs/probability_vs_length.png`: regression view of text length vs fraud probability.
- `experiments/figs/latency_throughput.png`: latency vs throughput benchmark; see `experiments/tables/benchmark_summary.csv`.

## Notes
All metrics computed on the frozen test split. Gray-zone policy maps probabilities within the band to human review.