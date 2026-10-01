"""Direct TF-IDF + logistic-regression baseline, outside the agent workspace."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).parent
task = ROOT / "task"
train = pd.read_csv(task / "train.csv")
test = pd.read_csv(task / "test.csv")
model = Pipeline(
    [
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=500, C=4.0, random_state=0)),
    ]
)
model.fit(train["text"], train["author"])
probability = model.predict_proba(test["text"])
output = pd.DataFrame(probability, columns=model.classes_)
output.insert(0, "id", test["id"].to_numpy())
output[["id", "EAP", "HPL", "MWS"]].to_csv(ROOT / "baseline_predictions.csv", index=False)
print(f"wrote {len(output)} predictions")
