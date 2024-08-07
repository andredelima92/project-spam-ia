import joblib
import os
from sklearn.metrics import accuracy_score, confusion_matrix
from data_preprocessing import load_and_preprocess_data
from train_model import train_and_save_model


def load_or_train_model(data_path, model_path):
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = joblib.load(model_path)
        x_train, x_test, y_train, y_test = load_and_preprocess_data(data_path)
    else:
        print("Model not found. Training a new model...")
        model, x_test, y_test = train_and_save_model(data_path, model_path)
    return model, x_test, y_test


if __name__ == "__main__":
    data_path = 'spam.csv'
    model_path = 'random_forest_model.joblib'

    model, x_test, y_test = load_or_train_model(data_path, model_path)

    # Faz previsões
    predicts = model.predict(x_test)

    # Avalia as previsões
    accuracy = accuracy_score(y_test, predicts)
    conf_matrix = confusion_matrix(y_test, predicts)

    print(f'Accuracy: {accuracy}')
    print('Confusion Matrix:')
    print(conf_matrix)
