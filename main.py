import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Load spam.csv file
spam = pd.read_csv('spam.csv', encoding='utf-8')

print(spam.head())
print(spam.shape)
print(spam['Category'].value_counts())  # Count the number of ham and spam

predict = spam['Message']
label = spam['Category']

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(predict, label, test_size=0.3, random_state=42)
