import datetime
import pickle
from pathlib import Path

import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Churn_Modelling.csv"
FEATURE_COLUMNS = [
    "CreditScore", "Gender", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
    "Geography_France", "Geography_Germany", "Geography_Spain",
]


def train_model() -> None:
    data = pd.read_csv(DATA_PATH).drop(
        columns=["RowNumber", "CustomerId", "Surname"]
    )
    target = data.pop("Exited")

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), [
                "CreditScore", "Age", "Tenure", "Balance",
                "NumOfProducts", "HasCrCard", "IsActiveMember",
                "EstimatedSalary",
            ]),
            ("geography", OneHotEncoder(
                handle_unknown="ignore", sparse_output=False
            ), ["Geography"]),
            ("gender", OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            ), ["Gender"]),
        ],
    )
    features = preprocessor.fit_transform(data)
    feature_names = list(preprocessor.get_feature_names_out())
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )

    model = Sequential([
        Input(shape=(x_train.shape[1],)),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    log_dir = BASE_DIR / "logs" / "fit" / datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            TensorBoard(log_dir=str(log_dir)),
        ],
        verbose=1,
    )

    probabilities = model.predict(x_test, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(f"Test ROC AUC: {roc_auc_score(y_test, probabilities):.3f}")
    print(classification_report(y_test, predictions, zero_division=0))

    model.save(BASE_DIR / "model.h5")
    with open(BASE_DIR / "preprocessor.pkl", "wb") as file:
        pickle.dump(preprocessor, file)
    with open(BASE_DIR / "feature_columns.pkl", "wb") as file:
        pickle.dump(feature_names, file)
    print("Saved model.h5, preprocessor.pkl, and feature_columns.pkl")


if __name__ == "__main__":
    train_model()
