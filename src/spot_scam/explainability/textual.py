from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from spot_scam.utils.logging import configure_logging

logger = configure_logging(__name__)


def top_tfidf_terms(vectorizer, classifier, *, top_n: int = 25) -> Dict[str, pd.DataFrame]:
    """
    Extract the top positive and negative TF-IDF terms based on linear model coefficients.
    """
    if not hasattr(classifier, "coef_"):
        raise AttributeError("Classifier does not expose coefficients for interpretability.")

    feature_names = np.array(vectorizer.get_feature_names_out())
    coefficients = np.asarray(classifier.coef_).ravel()

    if coefficients.shape[0] != feature_names.shape[0]:
        aligned = min(coefficients.shape[0], feature_names.shape[0])
        logger.debug(
            "Aligning TF-IDF coefficients (%d) with feature names (%d); using first %d entries.",
            coefficients.shape[0],
            feature_names.shape[0],
            aligned,
        )
        coefficients = coefficients[:aligned]
        feature_names = feature_names[:aligned]

    top_positive_idx = np.argsort(coefficients)[-top_n:][::-1]
    top_negative_idx = np.argsort(coefficients)[:top_n]

    pos_df = pd.DataFrame(
        {
            "term": feature_names[top_positive_idx],
            "weight": coefficients[top_positive_idx],
        }
    )
    neg_df = pd.DataFrame(
        {
            "term": feature_names[top_negative_idx],
            "weight": coefficients[top_negative_idx],
        }
    )
    return {"positive": pos_df, "negative": neg_df}


def token_frequency_analysis(
    df: pd.DataFrame,
    probabilities,
    *,
    top_k: int = 20,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Compute frequency differences of tokens between predicted frauds and legits.
    """
    import collections
    prob_series = pd.Series(probabilities, index=df.index)

    predicted_positive = df.loc[prob_series >= threshold, "text_all"]
    predicted_negative = df.loc[prob_series < threshold, "text_all"]

    positive_counter = collections.Counter()
    negative_counter = collections.Counter()

    for text in predicted_positive:
        positive_counter.update(text.split())
    for text in predicted_negative:
        negative_counter.update(text.split())

    tokens = set(list(dict(positive_counter.most_common(top_k)).keys()) + list(dict(negative_counter.most_common(top_k)).keys()))
    records = []
    for token in tokens:
        pos_freq = positive_counter[token]
        neg_freq = negative_counter[token]
        records.append(
            {
                "token": token,
                "positive_count": pos_freq,
                "negative_count": neg_freq,
                "difference": pos_freq - neg_freq,
            }
        )
    result = pd.DataFrame(records).sort_values(by="difference", ascending=False)
    return result


__all__ = ["top_tfidf_terms", "token_frequency_analysis"]
