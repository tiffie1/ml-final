import re

import nltk
from pandas import read_csv

stop_words = set(nltk.corpus.stopwords.words("english"))
lemmatizer = nltk.stem.WordNetLemmatizer()

# Uncomment if needed.
# nltk.download("stopwords")
# nltk.download("wordnet")


def clean_text(text: str) -> str:
    """
    Cleans text by applying several pre-processing steps:

        - Make all characters lowercase.
        - Remove non-ASCII characters (invalid characters).
        - Remove URLs, mentions, hashtags, and punctuation.
        - Lemmatize every word and remove stop words.
    """
    text = text.lower()
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # ASCII
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # URLs
    text = re.sub(r"\@\w+|\#", "", text)  # mentions, hashtags
    text = re.sub(r"[^\w\s]", "", text)  # punctuation

    # lemmatize & stop words
    text = " ".join(
        [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    )

    return text


if __name__ == "__main__":
    # columns = 'id', 'label', 'tweet'
    data = read_csv("./data/twitter.csv")
    data = data.dropna(subset=["tweet"])

    if data is not None:
        data["tweet"] = data["tweet"].apply(clean_text)
        data = data.dropna(subset=["tweet"])
        data.to_csv("./data/twitter_clean.csv")
