import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_feature_pipeline(categorical_cols, numeric_cols, tfidf_params=None):
    """
    Creates a pre-processing pipeline for crowdfunding data.
    """
    if tfidf_params is None:
        tfidf_params = {"ngram_range": (2, 2), "stop_words": "english", "max_features": 400}

    # Pipeline for categorical and numeric features
    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            (
                "num",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
                ),
                numeric_cols,
            ),
        ]
    )

    return preprocessor, TfidfVectorizer(**tfidf_params)


def process_features(df, preprocessor, tf_vectorizer, is_train=True):
    """
    Applies the pre-processing pipeline to the dataframe.
    """
    X_other = df.drop(columns=["blurb", "state"], errors="ignore")
    X_blurb = df["blurb"].astype(str)

    if is_train:
        X_transformed = preprocessor.fit_transform(X_other)
        tfidf_matrix = tf_vectorizer.fit_transform(X_blurb)
    else:
        X_transformed = preprocessor.transform(X_other)
        tfidf_matrix = tf_vectorizer.transform(X_blurb)

    # Combine dense and sparse features
    # Note: In the original notebook, they were converted to a single dense dataframe
    onehot_names = preprocessor.named_transformers_["cat"].get_feature_names_out()
    numeric_names = preprocessor.transformers_[1][2]
    tfidf_names = tf_vectorizer.get_feature_names_out()

    all_names = np.concatenate([onehot_names, numeric_names, tfidf_names])

    # Create final dense matrix (Caution: memory intensive for full dataset)
    X_final = np.hstack([X_transformed, tfidf_matrix.toarray()])

    return pd.DataFrame(X_final, columns=all_names)
