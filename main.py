import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load spam.csv file
spam = pd.read_csv('spam.csv', encoding='utf-8')

print(spam.head())
print(spam.shape)
print(spam['Category'].value_counts())  # Count the number of ham and spam

messages = spam['Message']
labels = spam['Category']

vector = TfidfVectorizer()
transformer = vector.fit_transform(messages)

x_train, x_test, y_train, y_test = train_test_split(
    transformer, labels, test_size=0.3)

florest = RandomForestClassifier(n_estimators=500)
florest.fit(x_train, y_train)

joblib.dump(florest, 'random_forest_model.joblib')