import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def load_and_preprocess_data(filepath, test_size=0.3):
    spam = pd.read_csv(filepath, encoding='utf-8')

    messages = spam['Message']
    labels = spam['Category']

    vector = TfidfVectorizer()
    transformer = vector.fit_transform(messages)

    x_train, x_test, y_train, y_test = train_test_split(
        transformer, labels, test_size=test_size)

    return x_train, x_test, y_train, y_test
