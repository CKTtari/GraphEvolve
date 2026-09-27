"""External direct baseline used only to validate the benchmark runner."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main() -> None:
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    target = train.pop("target")
    categorical = train.select_dtypes(include=["object"]).columns.tolist()
    numeric = [column for column in train.columns if column not in categorical]
    preprocessing = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", StandardScaler(), numeric),
        ]
    )
    model = Pipeline(
        [
            ("preprocessing", preprocessing),
            ("classifier", LogisticRegression(max_iter=2000)),
        ]
    )
    model.fit(train, target)
    pd.DataFrame({"prediction": model.predict(test)}).to_csv(
        "predictions.csv", index=False
    )


if __name__ == "__main__":
    main()
