from spot_scam.policy.gray_zone import classify_probability


def test_gray_zone_classification():
    policy = {
        "threshold": 0.6,
        "width": 0.2,
        "positive_label": "fraud",
        "negative_label": "legit",
        "review_label": "review",
    }
    assert classify_probability(0.9, **policy) == "fraud"
    assert classify_probability(0.1, **policy) == "legit"
    assert classify_probability(0.58, **policy) == "review"
