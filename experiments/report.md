# Spot the Scam Report — linear_svm_C1.0

## Configuration Snapshot
- Model: **linear_svm_C1.0** (classical)
- Calibration: isotonic
- Decision threshold: 0.4963
- Gray-zone width: 0.1

## Metrics Overview
| split      |       f1 |   precision |   recall |   roc_auc |   pr_auc |      brier |
|:-----------|---------:|------------:|---------:|----------:|---------:|-----------:|
| validation | 0.904762 |    0.92233  | 0.88785  |  0.994185 | 0.933485 | 0.00695038 |
| test       | 0.818653 |    0.918605 | 0.738318 |  0.973553 | 0.829869 | 0.0134569  |

## Token Signals (Top 20)
| token      |   positive_count |   negative_count |   difference |
|:-----------|-----------------:|-----------------:|-------------:|
| project    |               65 |              995 |         -930 |
| other      |               68 |             1536 |        -1468 |
| &          |               92 |             2039 |        -1947 |
| by         |               60 |             2477 |        -2417 |
| from       |              101 |             2529 |        -2428 |
| all        |              115 |             2661 |        -2546 |
| at         |               65 |             2749 |        -2684 |
| business   |               37 |             2747 |        -2710 |
| your       |              138 |             3157 |        -3019 |
| team       |               60 |             3532 |        -3472 |
| have       |               81 |             3681 |        -3600 |
| experience |              127 |             3938 |        -3811 |
| or         |              104 |             4216 |        -4112 |
| an         |               90 |             4211 |        -4121 |
| work       |              160 |             4313 |        -4153 |
| will       |              103 |             4575 |        -4472 |
| that       |               96 |             4674 |        -4578 |
| on         |              135 |             5123 |        -4988 |
| be         |              125 |             5242 |        -5117 |
| as         |              117 |             5357 |        -5240 |

## Slice Analysis
| slice               | category                            |   count |       f1 |   precision |   recall |    roc_auc |   pr_auc |       brier |
|:--------------------|:------------------------------------|--------:|---------:|------------:|---------:|-----------:|---------:|------------:|
| employment_type     | <missing>                           |     489 | 0.76     |    0.863636 | 0.678571 |   0.959947 | 0.815051 | 0.0208386   |
| employment_type     | Contract                            |     130 | 0.461538 |    0.75     | 0.333333 |   0.811295 | 0.433166 | 0.0514854   |
| employment_type     | Full-time                           |    1578 | 0.886792 |    0.959184 | 0.824561 |   0.994596 | 0.891705 | 0.00758704  |
| employment_type     | Part-time                           |     113 | 0.888889 |    1        | 0.8      |   0.992233 | 0.917647 | 0.0160155   |
| employment_type     | Temporary                           |      42 | 0        |    0        | 0        | nan        | 0        | 0.00897685  |
| required_experience | <missing>                           |     913 | 0.782609 |    0.9      | 0.692308 |   0.974482 | 0.836775 | 0.0192013   |
| required_experience | Associate                           |     317 | 0.666667 |    1        | 0.5      |   0.996032 | 0.642857 | 0.00277759  |
| required_experience | Director                            |      63 | 0        |    0        | 0        |   0.919355 | 0.125    | 0.0155983   |
| required_experience | Entry level                         |     333 | 0.952381 |    0.909091 | 1        |   0.997684 | 0.93411  | 0.00648761  |
| required_experience | Internship                          |      59 | 0.666667 |    1        | 0.5      |   0.934211 | 0.576923 | 0.0179504   |
| required_experience | Mid-Senior level                    |     525 | 0.820513 |    1        | 0.695652 |   0.952235 | 0.796139 | 0.0127404   |
| required_experience | Not Applicable                      |     148 | 0.571429 |    0.666667 | 0.5      |   0.96441  | 0.642857 | 0.018465    |
| required_education  | <missing>                           |    1151 | 0.778947 |    0.902439 | 0.685185 |   0.958869 | 0.794443 | 0.0164315   |
| required_education  | Associate Degree                    |      52 | 0        |    0        | 0        | nan        | 0        | 9.27255e-05 |
| required_education  | Bachelor's Degree                   |     630 | 0.857143 |    1        | 0.75     |   0.986393 | 0.854942 | 0.00729978  |
| required_education  | High School or equivalent           |     243 | 0.740741 |    0.769231 | 0.714286 |   0.973331 | 0.740227 | 0.0278847   |
| required_education  | Master's Degree                     |      69 | 1        |    1        | 1        |   1        | 1        | 0.000102462 |
| required_education  | Unspecified                         |     172 | 0.8      |    1        | 0.666667 |   0.994478 | 0.862637 | 0.00998505  |
| industry            | <missing>                           |     678 | 0.761905 |    0.857143 | 0.685714 |   0.96072  | 0.808566 | 0.0184158   |
| industry            | Computer Software                   |     178 | 0        |    0        | 0        |   0.991525 | 0.25     | 0.00503659  |
| industry            | Consumer Services                   |      35 | 1        |    1        | 1        |   1        | 1        | 0.000643622 |
| industry            | Financial Services                  |      93 | 0.666667 |    1        | 0.5      |   0.983516 | 0.666667 | 0.00988641  |
| industry            | Hospital & Health Care              |      64 | 0.857143 |    1        | 0.75     |   1        | 1        | 0.034564    |
| industry            | Information Technology and Services |     251 | 0.333333 |    0.666667 | 0.222222 |   0.89371  | 0.355501 | 0.0304438   |
| industry            | Internet                            |     154 | 0        |    0        | 0        | nan        | 0        | 9.79138e-05 |
| industry            | Marketing and Advertising           |     114 | 0.888889 |    1        | 0.8      |   0.988991 | 0.876923 | 0.00873488  |
| industry            | Oil & Energy                        |      40 | 1        |    1        | 1        |   1        | 1        | 0.00216085  |
| industry            | Telecommunications                  |      44 | 1        |    1        | 1        |   1        | 1        | 0.000244172 |
| function            | <missing>                           |     831 | 0.805556 |    0.90625  | 0.725    |   0.966166 | 0.824783 | 0.0153898   |
| function            | Accounting/Auditing                 |      35 | 0.444444 |    0.5      | 0.4      |   0.916667 | 0.548485 | 0.128374    |
| function            | Administrative                      |      80 | 0.967742 |    1        | 0.9375   |   0.997559 | 0.9875   | 0.0134171   |
| function            | Business Development                |      37 | 0        |    0        | 0        |   0.875    | 0.142857 | 0.0270089   |
| function            | Customer Service                    |     141 | 1        |    1        | 1        |   1        | 1        | 0.000809262 |
| function            | Design                              |      48 | 0        |    0        | 0        | nan        | 0        | 8.51203e-06 |
| function            | Engineering                         |     208 | 0.967742 |    1        | 0.9375   |   0.998861 | 0.985119 | 0.00444352  |
| function            | Health Care Provider                |      50 | 0        |    0        | 0        |   0.989796 | 0.5      | 0.0159676   |
| function            | Human Resources                     |      32 | 0.857143 |    0.75     | 1        |   1        | 1        | 0.0112632   |
| function            | Information Technology              |     258 | 0.625    |    1        | 0.454545 |   0.911299 | 0.588292 | 0.0227657   |
| function            | Management                          |      40 | 0        |    0        | 0        | nan        | 0        | 0.000356857 |
| function            | Marketing                           |     117 | 0.666667 |    1        | 0.5      |   0.976087 | 0.6      | 0.00848372  |
| function            | Other                               |      32 | 1        |    1        | 1        |   1        | 1        | 0.000359885 |
| function            | Sales                               |     186 | 0.666667 |    1        | 0.5      |   0.97962  | 0.576923 | 0.00519923  |

## Additional Visuals
- `experiments/figs/score_distribution_test.png`: probability density by class.
- `experiments/figs/threshold_sweep_val.png`: precision/recall/F1 trade-offs across thresholds.
- `experiments/figs/probability_vs_length.png`: regression view of text length vs fraud probability.
- `experiments/figs/latency_throughput.png`: latency vs throughput benchmark; see `experiments/tables/benchmark_summary.csv`.

## Notes
All metrics computed on the frozen test split. Gray-zone policy maps probabilities within the band to human review.