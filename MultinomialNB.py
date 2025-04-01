import warnings; warnings.filterwarnings('ignore')
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score

from Preprocessing import clean_text

if __name__ == "__main__":
    data = read_csv("./data/twitter.csv")
  
    train_dev_set, test_set = train_test_split(data, test_size = 0.2, random_state = 42)
    train_set, dev_set = train_test_split(train_dev_set, test_size = 0.25, random_state = 42)
    
    train_size = train_set.shape[0]
    dev_size = dev_set.shape[0]
    test_size = test_set.shape[0]
    total_size = data.shape[0]
    
    print(f"Total data points: {total_size}")
    print(f"Training set size: {train_size} ({train_size / total_size * 100:.2f}%)")
    print(f"Development set size: {dev_size} ({dev_size / total_size * 100:.2f}%)")
    print(f"Test set size: {test_size} ({test_size / total_size * 100:.2f}%)")
        
    train_set.loc[:, 'tweet'] = train_set['tweet'].apply(clean_text)
    dev_set.loc[:, 'tweet'] = dev_set['tweet'].apply(clean_text)
    test_set.loc[:, 'tweet'] = test_set['tweet'].apply(clean_text)
    
    vectorizer = CountVectorizer(max_features=10000)
    x_train = vectorizer.fit_transform(train_set["tweet"])
    x_dev = vectorizer.transform(dev_set["tweet"])
    x_test = vectorizer.transform(test_set["tweet"])

    
    x_train = x_train.toarray()
    x_dev = x_dev.toarray()
    x_test = x_test.toarray()

    
    y_train = train_set['label']
    y_dev = dev_set['label']
    y_test = test_set['label']
    
    
    #Naive Bayes Model
    model = MultinomialNB()
    model.fit(x_train, y_train)
    
    
    # Accuracy plot for train and dev sets
    train_score = model.score(x_train, y_train)
    dev_score = model.score(x_dev, y_dev)

    # Plot accuracy comparison
    plt.figure(figsize=(8, 5))
    scores = [train_score, dev_score]
    sets = ['Train', 'Dev']

    plt.bar(sets, scores, color=['blue', 'orange'])
    plt.title("Accuracy of Multinomial Naive Bayes on Train and Dev Sets")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)  # Accuracy range is from 0 to 1
    plt.show()
    
    y_pred = model.predict(x_test)
    
    # Print evaluation metrics
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))


