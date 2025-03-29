from typing import cast

import numpy as np
import pandas as pd
from pandas import DataFrame, read_csv
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

from GraphicsUtils import bar_binary_plot

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

    X_test = vectorizer.transform(data_test["tweet"])  # type: ignore[attr-defined]
    X_test = X_test.toarray()
    X_test = np.array(X_test)

    y_train = data_train["label"]
    y_dev = data_dev["label"]
    y_test_true = data_test["label"]


    model = GaussianMixture(2)

    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    bar_binary_plot(pd.Series(y_test_pred), "pred id distribution", "id")
