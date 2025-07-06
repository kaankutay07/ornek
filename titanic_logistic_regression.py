"""Simple logistic regression example using the Titanic dataset.

This script is intended for educational purposes. It loads the Titanic dataset
from the seaborn library, performs minimal preprocessing and trains a logistic
regression model to predict passenger survival.
"""

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_and_prepare():
    """Load the Titanic dataset and prepare features and labels."""
    df = sns.load_dataset("titanic")

    # Select a few informative columns
    cols = [
        "survived",
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "alone",
    ]
    df = df[cols].dropna()

    # Convert categorical features to numeric
    df = pd.get_dummies(df, columns=["sex", "alone"], drop_first=True)

    X = df.drop("survived", axis=1)
    y = df["survived"]
    return X, y


def train_and_evaluate(X, y):
    """Train a logistic regression model and print accuracy."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.3f}")


if __name__ == "__main__":
    features, labels = load_and_prepare()
    train_and_evaluate(features, labels)
