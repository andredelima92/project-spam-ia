import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

spam = pd.read_csv('spam.csv', encoding='utf-8')

messages = spam['Message']
labels = spam['Category']

vector = TfidfVectorizer()
transformer = vector.fit_transform(messages)

x_train, x_test, y_train, y_test = train_test_split(transformer, labels, test_size=0.3)

# Carrega o modelo salvo
florest = joblib.load('random_forest_model.joblib')

# Faz previsões
predicts = florest.predict(x_test)

# Avalia as previsões
accuracy = accuracy_score(y_test, predicts)
conf_matrix = confusion_matrix(y_test, predicts)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
