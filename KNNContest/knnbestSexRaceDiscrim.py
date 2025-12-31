import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
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

# Preprocess
pre = ColumnTransformer(
    transformers=[
        ("num", MinMaxScaler(feature_range=(0, 2)), num_cols),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop",
)
X_train = pre.fit_transform(Xtr_df)  
X_dev   = pre.transform(Xdv_df)

# k-NN: k=41, Manhattan only
clf = KNeighborsClassifier(n_neighbors=41, metric="minkowski", p=1, n_jobs=-1)  # L1
clf.fit(X_train, y_train)

y_train_pred = clf.predict(X_train)
y_dev_pred   = clf.predict(X_dev)

def err_and_pos(y_true, y_pred):
    err = 100.0 * (1.0 - accuracy_score(y_true, y_pred))
    pos = 100.0 * (y_pred == 1).mean()
    return err, pos

tr_err, tr_pos = err_and_pos(y_train, y_train_pred)
dv_err, dv_pos = err_and_pos(y_dev,   y_dev_pred)
print(f"[L1,k=41] train_err {tr_err:.2f}% (+:{tr_pos:.2f}%)  dev_err {dv_err:.2f}% (+:{dv_pos:.2f}%)")


# Group metrics: true vs predicted positive rates by sex/race
dev_df = dev.copy()
dev_df["_y_true"] = (dev_df["target"] == ">50K").astype(int)
dev_df["_y_pred"] = y_dev_pred

def rates_by(col):
    out = (dev_df.groupby(col)
           .agg(true_pos_rate=("_y_true","mean"),
                pred_pos_rate=("_y_pred","mean"),
                count=("id","size")))
    return (out * 100).round(2).sort_values("count", ascending=False)

print("\nBy sex:\n", rates_by("sex"))
print("\nBy race:\n", rates_by("race"))
