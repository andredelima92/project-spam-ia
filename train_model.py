import joblib
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import load_and_preprocess_data

def train_and_save_model(data_path, model_path):
    x_train, x_test, y_train, y_test = load_and_preprocess_data(data_path)

    florest = RandomForestClassifier(n_estimators=500)
    florest.fit(x_train, y_train)

    joblib.dump(florest, model_path)
    return florest, x_test, y_test

if __name__ == "__main__":
    data_path = 'spam.csv'
    model_path = 'random_forest_model.joblib'
    train_and_save_model(data_path, model_path)