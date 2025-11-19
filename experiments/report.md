# Spot the Scam Report - ensemble_top3

## Configuration Snapshot
- Model: **ensemble_top3** (classical)
- Calibration: none
- Decision threshold: 0.5802
- Gray-zone width: 0.1

## Metrics Overview
| split      |       f1 |   precision |   recall |   roc_auc |   pr_auc |    brier |
|:-----------|---------:|------------:|---------:|----------:|---------:|---------:|
| validation | 0.856115 |    0.929688 | 0.793333 |  0.988983 | 0.905253 | 0.010259 |
| test       | 0.772059 |    0.853659 | 0.704698 |  0.986342 | 0.865872 | 0.014332 |

## Token Signals (Top 20)
| token      |   positive_count |   negative_count |   difference |
|:-----------|-----------------:|-----------------:|-------------:|
| solutions  |               92 |             1121 |        -1029 |
| -          |               88 |             1957 |        -1869 |
| &          |              109 |             1993 |        -1884 |
| customer   |               93 |             2167 |        -2074 |
| this       |               88 |             2416 |        -2328 |
| from       |              121 |             2572 |        -2451 |
| by         |               79 |             2537 |        -2458 |
| business   |               74 |             2596 |        -2522 |
| all        |              100 |             2948 |        -2848 |
| at         |               69 |             2986 |        -2917 |
| your       |              107 |             3360 |        -3253 |
| team       |               84 |             3568 |        -3484 |
| have       |               77 |             4018 |        -3941 |
| experience |              125 |             4521 |        -4396 |
| an         |              116 |             4521 |        -4405 |
| work       |              145 |             4683 |        -4538 |
| will       |               82 |             4645 |        -4563 |
| that       |              116 |             4697 |        -4581 |
| or         |              191 |             4944 |        -4753 |
| on         |              189 |             5490 |        -5301 |

## Slice Analysis
| slice    | category               |   count |       f1 |   precision |   recall |    roc_auc |   pr_auc |       brier |
|:---------|:-----------------------|--------:|---------:|------------:|---------:|-----------:|---------:|------------:|
| function | <missing>              |     780 | 0.888889 |    0.933333 | 0.848485 |   0.995619 | 0.942781 | 0.00729904  |
| function | Administrative         |     111 | 0.882353 |    0.9375   | 0.833333 |   0.993429 | 0.971655 | 0.0250434   |
| function | Business Development   |      34 | 1        |    1        | 1        |   1        | 1        | 2.96929e-05 |
| function | Customer Service       |     153 | 0.8      |    1        | 0.666667 |   0.98818  | 0.901144 | 0.0216835   |
| function | Design                 |      62 | 0        |    0        | 0        |   1        | 1        | 0.0120521   |
| function | Engineering            |     226 | 0.893617 |    1        | 0.807692 |   0.999038 | 0.993172 | 0.0159782   |
| function | Health Care Provider   |      53 | 0        |    0        | 0        | nan        | 0        | 0.0286519   |
| function | Human Resources        |      40 | 1        |    1        | 1        |   1        | 1        | 2.58352e-05 |
| function | Information Technology |     294 | 0.4      |    1        | 0.25     |   0.946121 | 0.4916   | 0.00919455  |
| function | Management             |      52 | 0        |    0        | 0        |   0.980392 | 0.5      | 0.0183585   |
| function | Marketing              |     146 | 1        |    1        | 1        |   1        | 1        | 0.000484177 |
| function | Not Mentioned          |     717 | 0.622951 |    0.703704 | 0.558824 |   0.958445 | 0.717654 | 0.0242817   |
| function | Other                  |      56 | 0.666667 |    1        | 0.5      |   0.990741 | 0.833333 | 0.0161148   |
| function | Sales                  |     254 | 0.666667 |    0.571429 | 0.8      |   0.984337 | 0.834483 | 0.010143    |

## Additional Visuals
- `experiments/figs/score_distribution_test.png`: probability density by class.
- `experiments/figs/threshold_sweep_val.png`: precision/recall/F1 trade-offs across thresholds.
- `experiments/figs/probability_vs_length.png`: regression view of text length vs fraud probability.
- `experiments/figs/latency_throughput.png`: latency vs throughput benchmark; see `experiments/tables/benchmark_summary.csv`.

## Notes
All metrics computed on the frozen test split. Gray-zone policy maps probabilities within the band to human review.