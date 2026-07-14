# flake8: noqa: E501
"""
Solucion del laboratorio: modelo de default de tarjetas de credito.
"""

import gzip
import json
import os
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def _clean(df):
    df = df.copy()
    df = df.rename(columns={"default payment next month": "default"})
    df = df.drop(columns=["ID"])
    df = df.dropna()
    df = df[(df["EDUCATION"] != 0) & (df["MARRIAGE"] != 0)]
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)
    return df


def pregunta_01():
    """Ejecuta todo el flujo del laboratorio."""

    # Paso 1: cargar y limpiar
    train = _clean(pd.read_csv("files/input/train_data.csv.zip"))
    test = _clean(pd.read_csv("files/input/test_data.csv.zip"))

    # Paso 2: dividir
    x_train, y_train = train.drop(columns=["default"]), train["default"]
    x_test, y_test = test.drop(columns=["default"]), test["default"]

    # Paso 3: pipeline
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="passthrough",
    )
    pipeline = Pipeline(
        steps=[
            ("OneHotEncoder", preprocessor),
            (
                "RandomForestClassifier",
                RandomForestClassifier(random_state=42, n_jobs=-1),
            ),
        ]
    )

    # Paso 4: optimizacion de hiperparametros
    param_grid = {
        "RandomForestClassifier__n_estimators": [300],
        "RandomForestClassifier__max_depth": [20],
        "RandomForestClassifier__min_samples_leaf": [1],
    }
    model = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=1,
    )
    model.fit(x_train, y_train)

    # Paso 5: guardar modelo comprimido
    os.makedirs("files/models", exist_ok=True)
    with gzip.open("files/models/model.pkl.gz", "wb") as file:
        pickle.dump(model, file)

    # Paso 6 y 7: metricas y matrices de confusion
    os.makedirs("files/output", exist_ok=True)
    results = []
    for dataset_name, x, y in [("train", x_train, y_train), ("test", x_test, y_test)]:
        y_pred = model.predict(x)
        results.append(
            {
                "type": "metrics",
                "dataset": dataset_name,
                "precision": precision_score(y, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y, y_pred),
                "recall": recall_score(y, y_pred),
                "f1_score": f1_score(y, y_pred),
            }
        )

    for dataset_name, x, y in [("train", x_train, y_train), ("test", x_test, y_test)]:
        y_pred = model.predict(x)
        cm = confusion_matrix(y, y_pred)
        results.append(
            {
                "type": "cm_matrix",
                "dataset": dataset_name,
                "true_0": {
                    "predicted_0": int(cm[0][0]),
                    "predicted_1": int(cm[0][1]),
                },
                "true_1": {
                    "predicted_0": int(cm[1][0]),
                    "predicted_1": int(cm[1][1]),
                },
            }
        )

    with open("files/output/metrics.json", "w", encoding="utf-8") as file:
        for row in results:
            file.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    pregunta_01()