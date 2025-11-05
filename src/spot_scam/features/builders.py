from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy import sparse

from spot_scam.features.tabular import create_tabular_features
from spot_scam.features.text import build_tfidf_vectorizer


@dataclass
class FeatureBundle:
    tfidf_vectorizer: any
    tfidf_train: sparse.csr_matrix
    tfidf_val: sparse.csr_matrix
    tfidf_test: sparse.csr_matrix
    tabular_scaler: any
    tabular_train: sparse.csr_matrix
    tabular_val: sparse.csr_matrix
    tabular_test: sparse.csr_matrix
    feature_names: Tuple[str, ...]


def build_feature_bundle(train_df, val_df, test_df, config: Dict) -> FeatureBundle:
    """Construct TF-IDF and tabular feature matrices for classical models."""
    vectorizer, _ = build_tfidf_vectorizer(config)
    tfidf_train = vectorizer.fit_transform(train_df["text_all"])
    tfidf_val = vectorizer.transform(val_df["text_all"])
    tfidf_test = vectorizer.transform(test_df["text_all"])

    tab_train_df = create_tabular_features(train_df, config)
    tab_val_df = create_tabular_features(val_df, config)
    tab_test_df = create_tabular_features(test_df, config)

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    tab_train = scaler.fit_transform(tab_train_df.astype(np.float32))
    tab_val = scaler.transform(tab_val_df.astype(np.float32))
    tab_test = scaler.transform(tab_test_df.astype(np.float32))

    tab_train_sparse = sparse.csr_matrix(tab_train)
    tab_val_sparse = sparse.csr_matrix(tab_val)
    tab_test_sparse = sparse.csr_matrix(tab_test)

    feature_names = tuple(tab_train_df.columns.tolist())

    return FeatureBundle(
        tfidf_vectorizer=vectorizer,
        tfidf_train=tfidf_train,
        tfidf_val=tfidf_val,
        tfidf_test=tfidf_test,
        tabular_scaler=scaler,
        tabular_train=tab_train_sparse,
        tabular_val=tab_val_sparse,
        tabular_test=tab_test_sparse,
        feature_names=feature_names,
    )


__all__ = ["FeatureBundle", "build_feature_bundle"]

