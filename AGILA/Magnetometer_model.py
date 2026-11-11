"""
Sources: https://www.youtube.com/watch?v=kFwe2ZZU7yw
"""

from random import random
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from RandomForest import RandomForest

data = datasets.load_breast_cancer()
X = data.data
y = data.target

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

df = pd.read_csv("synthetic_bno055_magnetometer_data.csv")   # same folder as this script

def wrap180(x):
    return (x + 180) % 360 - 180

def window_features(w):
    m = w[["mx", "my", "mz"]].values
    a = w[["ax", "ay", "az"]].values
    mn = np.linalg.norm(m, axis=1)
    dip = np.degrees(np.arccos(np.clip(
        (m * a).sum(1) / (mn * np.linalg.norm(a, axis=1) + 1e-9), -1, 1)))
    d_fused = wrap180(w["heading"].iloc[-1] - w["heading"].iloc[0])
    d_gyro = -np.sum(w["gz"].values) * 0.01          # 100 Hz, gz in deg/s
    return [mn.mean(), mn.std(), np.abs(np.diff(mn)).max(),
            dip.mean(), dip.std(), abs(d_fused - d_gyro),
            w["cm"].iloc[-1], w["cs"].iloc[-1]]

WIN, STEP = 150, 75            # 1.5 s windows, 50% overlap
rows, labels, groups = [], [], []
for sid, s in df.groupby("session_id"):
    s = s.reset_index(drop=True)
    for start in range(0, len(s) - WIN + 1, STEP):
        w = s.iloc[start:start + WIN]
        rows.append(window_features(w))
        labels.append(int(w["anomaly_label"].max()))   # 1 = disturbed window
        groups.append(sid)

X = np.array(rows)
y = np.array(labels)
groups = np.array(groups)
print("windows:", X.shape, "anomaly share:", y.mean().round(3))

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=1234)
train_idx, test_idx = next(gss.split(X, y, groups=groups))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

clf = RandomForest(n_trees=20)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

acc =  accuracy(y_test, predictions)
print(acc)

print("baseline (always normal):", np.mean(y_test == 0).round(3))
tp = np.sum((predictions == 1) & (y_test == 1))
fn = np.sum((predictions == 0) & (y_test == 1))
fp = np.sum((predictions == 1) & (y_test == 0))
print("anomaly recall:", round(tp / max(tp + fn, 1), 3), " false alarms:", fp)