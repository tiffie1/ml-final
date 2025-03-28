from typing import cast

from pandas import DataFrame, read_csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

from Preprocessing import clean_text

if __name__ == "__main__":
    # columns = 'id', 'label', 'tweet'
    data = read_csv("./data/twitter.csv")
    data = data.set_index("id")

    data["tweet"] = data["tweet"].apply(clean_text)

    # X_train.to_csv("./data/twitter_clean.csv")

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

    X_train = vectorizer.fit_transform(data_train["tweet"])
    X_dev = vectorizer.fit_transform(data_dev["tweet"])
    X_test = vectorizer.fit_transform(data_test["tweet"])
    y_train = data_train["id"]
    y_dev = data_dev["id"]
    y_test_true = data_test["id"]
