"""Fixed no-agent word+character TF-IDF baseline for the insults task."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

ROOT = Path(__file__).parent
task = ROOT / "task"
train = pd.read_csv(task / "train.csv")
test = pd.read_csv(task / "test.csv")

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
train_features = features.fit_transform(train["Comment"].fillna("").astype(str))
test_features = features.transform(test["Comment"].fillna("").astype(str))
model = LogisticRegression(C=4.0, max_iter=1000, solver="liblinear", random_state=42)
model.fit(train_features, train["Insult"].astype(int))
probabilities = model.predict_proba(test_features)[:, 1]
output = test.copy()
output["Insult"] = probabilities
output = output[["Insult", "Date", "Comment"]]
output.to_csv(ROOT / "baseline_predictions.csv", index=False)
print(f"wrote {len(output)} fixed word+character predictions")
