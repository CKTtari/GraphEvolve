"""Fixed no-agent comparator for the accepted word+character representation."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion


ROOT = Path(__file__).parent
task = ROOT / "task"
train = pd.read_csv(task / "train.csv")
test = pd.read_csv(task / "test.csv")
sample = pd.read_csv(task / "sample_submission.csv")
labels = train["author"].astype(str)

features = FeatureUnion(
    [
        (
            "word",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
                max_features=250000,
            ),
        ),
        (
            "char",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                min_df=2,
                sublinear_tf=True,
                max_features=250000,
            ),
        ),
    ]
)
train_features = features.fit_transform(train["text"].fillna("").astype(str))
test_features = features.transform(test["text"].fillna("").astype(str))
model = LogisticRegression(C=4.0, max_iter=1000, solver="lbfgs", random_state=42)
model.fit(train_features, labels)
probabilities = model.predict_proba(test_features)
classes = list(model.classes_)
output = pd.DataFrame({"id": test["id"]})
for label in ["EAP", "HPL", "MWS"]:
    output[label] = probabilities[:, classes.index(label)]
output.to_csv(ROOT / "baseline_combined_predictions.csv", index=False)
print(f"wrote {len(output)} fixed word+character predictions")
