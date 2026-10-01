"""Fixed no-agent tabular baseline for NOMAD 2018."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).parent
task = ROOT / "task"
train = pd.read_csv(task / "train.csv")
test = pd.read_csv(task / "test.csv")
targets = ["formation_energy_ev_natom", "bandgap_energy_ev"]
features = [column for column in train.columns if column not in {"id", *targets}]
x_train = train[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
x_test = test[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
y_train = np.log1p(train[targets].to_numpy(dtype=float))
model = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
model.fit(x_train, y_train)
predictions = np.maximum(np.expm1(model.predict(x_test)), 0.0)
output = pd.DataFrame({"id": test["id"]})
for index, target in enumerate(targets):
    output[target] = predictions[:, index]
output.to_csv(ROOT / "baseline_predictions.csv", index=False)
print(f"wrote {len(output)} fixed tabular predictions")
