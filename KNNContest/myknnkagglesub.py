import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier

# Paths
train_path = "hw1-data/income.train.5k.csv"
dev_path   = "hw1-data/income.dev.csv"
test_path  = "hw1-data/income.test.blind.csv"

# Load
train = pd.read_csv(train_path, header=0, skipinitialspace=True, na_filter=False)
dev   = pd.read_csv(dev_path,   header=0, skipinitialspace=True, na_filter=False)
test  = pd.read_csv(test_path,  header=0, skipinitialspace=True, na_filter=False)

num_cols = ["age", "hours"]
cat_cols = ["sector", "edu", "marriage", "occupation", "race", "sex", "country"]
feature_cols = num_cols + cat_cols

Xtr_df = train[feature_cols].copy()
Xdv_df = dev[feature_cols].copy()
Xt_df  = test[feature_cols].copy()

y_train = (train["target"] == ">50K").astype(int).values
y_dev   = (dev["target"]   == ">50K").astype(int).values

# Smart preprocessor: scale numerics to [0,2], one-hot categoricals
pre = ColumnTransformer(
    transformers=[
        ("num", MinMaxScaler(feature_range=(0, 2)), num_cols),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop",
)

X_train = pre.fit_transform(Xtr_df)
X_dev   = pre.transform(Xdv_df)
X_test  = pre.transform(Xt_df)

# Best model from sweep
best_k = 41
clf = KNeighborsClassifier(n_neighbors=best_k, metric="minkowski", p=1, n_jobs=-1) 
clf.fit(X_train, y_train)

# Predict dev
y_dev_pred  = clf.predict(X_dev)
y_test_pred = clf.predict(X_test)

dev_err = 100 * (1 - (y_dev_pred == y_dev).mean())
dev_pos = 100 * (y_dev_pred == 1).mean()
print(f"Dev err: {dev_err:.2f}% | Dev +rate: {dev_pos:.2f}%")  

test_pos = 100 * (y_test_pred == 1).mean()
print(f"Test +rate: {test_pos:.2f}%") 

# Write required outputs
pred_str = np.where(y_test_pred == 1, ">50K", "<=50K")

# Same format as train/dev: append predicted
test_with_pred = test.copy()
test_with_pred["target"] = pred_str
test_with_pred.to_csv("income.test.predicted.csv", index=False)
print("Wrote income.test.predicted.csv")

# Kaggle-friendly file (id,target)
pd.DataFrame({"id": test["id"], "target": pred_str}).to_csv("income.test.kaggle.csv", index=False)
print("Wrote income.test.kaggle.csv")