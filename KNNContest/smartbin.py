import time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load
train = pd.read_csv("hw1-data/income.train.5k.csv", header=0, skipinitialspace=True, na_filter=False)
dev   = pd.read_csv("hw1-data/income.dev.csv",      header=0, skipinitialspace=True, na_filter=False)


num_cols = ["age", "hours"]
cat_cols = ["sector", "edu", "marriage", "occupation", "race", "sex", "country"]
feature_cols = num_cols + cat_cols

Xtr_df = train[feature_cols].copy()
Xdv_df = dev[feature_cols].copy()
y_train = (train["target"] == ">50K").astype(int).values
y_dev   = (dev["target"]   == ">50K").astype(int).values


#  Smart preprocessing
pre = ColumnTransformer(
    transformers=[
        ("num", MinMaxScaler(feature_range=(0, 1)), num_cols),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop",
)

X_train = pre.fit_transform(Xtr_df)
X_dev   = pre.transform(Xdv_df)

print("Smart-binarized shapes:")
print("  X_train:", X_train.shape, "  X_dev:", X_dev.shape)

# k-NN sweep (odd k = 1..99), Manhattan distance
def pos_rate(y_pred):
    return 100.0 * np.mean(y_pred == 1)

def err_rate(y_true, y_pred):
    return 100.0 * (1.0 - accuracy_score(y_true, y_pred))

rows = []

for k in range(1, 100, 2):
    clf = KNeighborsClassifier(n_neighbors=k, metric="minkowski", p=1, weights="uniform", n_jobs=-1)
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    t_fit = time.perf_counter() - t0

    t1 = time.perf_counter()
    yhat_tr = clf.predict(X_train)
    yhat_dv = clf.predict(X_dev)
    t_pred = time.perf_counter() - t1

    tr_err = err_rate(y_train, yhat_tr)
    dv_err = err_rate(y_dev,   yhat_dv)
    tr_pos = pos_rate(yhat_tr)
    dv_pos = pos_rate(yhat_dv)

    rows.append((k, tr_err, tr_pos, dv_err, dv_pos, t_fit, t_pred))

# Pretty print
for k, tr_err, tr_pos, dv_err, dv_pos, t_fit, t_pred in rows:
    print(f"k={k:2d} train_err {tr_err:5.1f}% (+:{tr_pos:5.1f}%)  dev_err {dv_err:5.1f}% (+:{dv_pos:5.1f}%)")

best_row = min(rows, key=lambda r: (r[3], r[0]))
best_k, _, _, best_dev_err, best_dev_pos, _, _ = best_row
print(f"\nBest dev error: {best_dev_err:.2f}% at k={best_k} (dev +rate {best_dev_pos:.1f}%)")
