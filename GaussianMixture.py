from typing import cast

import numpy as np
import pandas as pd
from pandas import DataFrame, Series, read_csv
from scipy.sparse import csr_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # columns = 'id', 'label', 'tweet'
    data = read_csv("./data/twitter_clean.csv")
    data = data.set_index("id")
    data = data.dropna(subset=["tweet"])

    # 60%, 20%, 20% split
    data_train, data_temp = cast(
        tuple[DataFrame, DataFrame],
        train_test_split(data, test_size=0.4, random_state=42),
    )
    data_dev, data_test = cast(
        tuple[DataFrame, DataFrame],
        train_test_split(data_temp, test_size=0.5, random_state=42),
    )

    vectorizer = CountVectorizer(binary=True, min_df=5)

    # Create bag of words feature matrices.
    X_train = vectorizer.fit_transform(data_train["tweet"]).toarray()  # type: ignore[attr-defined]
    X_train = np.array(X_train)

    X_dev: csr_matrix = vectorizer.transform(data_dev["tweet"])  # type: ignore[attr-defined]
    X_dev = X_dev.toarray()
    X_dev = np.array(X_dev)

    X_test: csr_matrix = vectorizer.transform(data_test["tweet"])  # type: ignore[attr-defined]
    X_test = X_test.toarray()
    X_test = np.array(X_test)

    y_train: Series = data_train["label"]
    y_dev: Series = data_dev["label"]
    y_test_true: Series = data_test["label"]


    # Train and predict with model.
    model = LinearDiscriminantAnalysis(solver="svd")
    model.fit(X_train, y_train)
    y_test_pred: Series = pd.Series(model.predict(X_test))

    print(y_test_true.value_counts())
    print(y_test_pred.value_counts(), "\n")
    print(classification_report(y_test_true, y_test_pred))
