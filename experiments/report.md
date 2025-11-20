# Spot the Scam Report - ensemble_top3

## Configuration Snapshot
- Model: **ensemble_top3** (classical)
- Calibration: none
- Decision threshold: 0.4137
- Gray-zone width: 0.1

## Metrics Overview
| split      |       f1 |   precision |   recall |   roc_auc |   pr_auc |     brier |
|:-----------|---------:|------------:|---------:|----------:|---------:|----------:|
| validation | 0.854093 |    0.916031 | 0.8      |  0.986725 | 0.892882 | 0.0107801 |
| test       | 0.766423 |    0.84     | 0.704698 |  0.983326 | 0.857636 | 0.0148685 |

## Token Signals (Top 20)
| token      |   positive_count |   negative_count |   difference |
|:-----------|-----------------:|-----------------:|-------------:|
| solutions  |               92 |             1121 |        -1029 |
| -          |               88 |             1957 |        -1869 |
| &          |              109 |             1993 |        -1884 |
| customer   |               95 |             2165 |        -2070 |
| this       |               88 |             2416 |        -2328 |
| from       |              121 |             2572 |        -2451 |
| new        |               55 |             2537 |        -2482 |
| business   |               74 |             2596 |        -2522 |
| all        |              100 |             2948 |        -2848 |
| at         |               69 |             2986 |        -2917 |
| your       |              107 |             3360 |        -3253 |
| team       |               84 |             3568 |        -3484 |
| have       |               77 |             4018 |        -3941 |
| experience |              126 |             4520 |        -4394 |
| an         |              116 |             4521 |        -4405 |
| work       |              146 |             4682 |        -4536 |
| will       |               82 |             4645 |        -4563 |
| that       |              116 |             4697 |        -4581 |
| or         |              191 |             4944 |        -4753 |
| on         |              189 |             5490 |        -5301 |

## Slice Analysis
| slice    | category               |   count |       f1 |   precision |   recall |    roc_auc |   pr_auc |       brier |
|:---------|:-----------------------|--------:|---------:|------------:|---------:|-----------:|---------:|------------:|
| function | <missing>              |     780 | 0.888889 |    0.933333 | 0.848485 |   0.994686 | 0.940225 | 0.00733224  |
| function | Administrative         |     111 | 0.882353 |    0.9375   | 0.833333 |   0.992832 | 0.968893 | 0.0267593   |
| function | Business Development   |      34 | 1        |    1        | 1        |   1        | 1        | 4.12444e-05 |
| function | Customer Service       |     153 | 0.761905 |    0.888889 | 0.666667 |   0.987589 | 0.899728 | 0.0228544   |
| function | Design                 |      62 | 0        |    0        | 0        |   1        | 1        | 0.0127206   |
| function | Engineering            |     226 | 0.893617 |    1        | 0.807692 |   0.999135 | 0.990323 | 0.0166655   |
| function | Health Care Provider   |      53 | 0        |    0        | 0        | nan        | 0        | 0.0262458   |
| function | Human Resources        |      40 | 1        |    1        | 1        |   1        | 1        | 3.67283e-05 |
| function | Information Technology |     294 | 0.333333 |    0.5      | 0.25     |   0.956897 | 0.504456 | 0.00954508  |
| function | Management             |      52 | 0        |    0        | 0        |   0.95098  | 0.25     | 0.0186006   |
| function | Marketing              |     146 | 1        |    1        | 1        |   1        | 1        | 0.000571932 |
| function | Not Mentioned          |     717 | 0.622951 |    0.703704 | 0.558824 |   0.942662 | 0.685389 | 0.026156    |
| function | Other                  |      56 | 0.666667 |    1        | 0.5      |   0.990741 | 0.833333 | 0.0159193   |
| function | Sales                  |     254 | 0.666667 |    0.571429 | 0.8      |   0.984337 | 0.837037 | 0.0110075   |

## Additional Visuals
- `experiments/figs/score_distribution_test.png`: probability density by class.
- `experiments/figs/threshold_sweep_val.png`: precision/recall/F1 trade-offs across thresholds.
- `experiments/figs/probability_vs_length.png`: regression view of text length vs fraud probability.
- `experiments/figs/latency_throughput.png`: latency vs throughput benchmark; see `experiments/tables/benchmark_summary.csv`.
