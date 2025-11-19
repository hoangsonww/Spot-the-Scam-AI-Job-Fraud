from __future__ import annotations

from typing import Dict, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(config: Dict) -> Tuple[TfidfVectorizer, Dict]:
    tfidf_conf = config["models"]["classical"]["tfidf"]
    vectorizer = TfidfVectorizer(
        ngram_range=tuple(tfidf_conf["ngram_range"]),
        min_df=tfidf_conf["min_df"],
        max_df=tfidf_conf["max_df"],
        sublinear_tf=tfidf_conf.get("sublinear_tf", True),
        lowercase=tfidf_conf.get("lowercase", True),
        max_features=config["preprocessing"].get("max_vocabulary_size"),
    )
    metadata = {
        "ngram_range": tfidf_conf["ngram_range"],
        "min_df": tfidf_conf["min_df"],
        "max_df": tfidf_conf["max_df"],
        "sublinear_tf": tfidf_conf.get("sublinear_tf", True),
    }
    return vectorizer, metadata


__all__ = ["build_tfidf_vectorizer"]
