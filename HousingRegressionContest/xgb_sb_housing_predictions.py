import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor


train_path = "my_train.csv"
dev_path   = "my_dev.csv"
full_train_path = "train.csv"
test_path  = "test.csv"
out_path   = "xgb_house_tuned_submission.csv"

# Tune XGB on my_train / my_dev with outlier removal

train = pd.read_csv(train_path)
dev   = pd.read_csv(dev_path)

# y, X for tuning
y_train_full = train["SalePrice"].values
y_dev        = dev["SalePrice"].values

Xtr_df_full = train.drop(columns=["Id", "SalePrice"])
Xdv_df      = dev.drop(columns=["Id", "SalePrice"])

# --- Outlier removal on my_train: 99th percentile on GrLivArea and SalePrice ---
q_gr = Xtr_df_full["GrLivArea"].quantile(0.99)
q_sp = pd.Series(y_train_full).quantile(0.99)

mask_inliers = (Xtr_df_full["GrLivArea"] <= q_gr) & (y_train_full <= q_sp)
Xtr_df = Xtr_df_full[mask_inliers].copy()
y_train = y_train_full[mask_inliers].copy()

print(f"Removed {len(y_train_full) - len(y_train)} outliers from my_train")

# --- Define categorical / numeric columns based on inlier training data ---
cat_cols = [c for c in Xtr_df.columns if Xtr_df[c].dtype == "object"]
if "MSSubClass" in Xtr_df.columns and "MSSubClass" not in cat_cols:
    cat_cols.append("MSSubClass")

num_cols = [c for c in Xtr_df.columns if c not in cat_cols]

# Ensure numeric columns are numeric in ALL splits
for df_ in (Xtr_df, Xdv_df):
    df_[num_cols] = df_[num_cols].apply(pd.to_numeric, errors="coerce")

# --- Preprocessing: no polynomial, just imputation + OHE ---
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

feats = ColumnTransformer(
    transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ]
)

# Transform features
Xtr = feats.fit_transform(Xtr_df)
Xdv = feats.transform(Xdv_df)

print(f"Train features shape (inliers): {Xtr.shape}")
print(f"Dev features shape:             {Xdv.shape}")

# Work in log-price space
y_train_log = np.log1p(y_train)
y_dev_log   = np.log1p(y_dev)

# --- Fixed tuned hyperparameters ---
best_params = {
    "learning_rate":   0.03,
    "max_depth":       2,
    "min_child_weight": 3,
    "subsample":       0.7,
    "colsample_bytree": 0.7,
    "gamma":           0.0,
    "reg_lambda":      10.0,
    "reg_alpha":       0.1,
    "n_estimators":    2400,
}

print("=== XGB evaluation on my_train/my_dev with fixed tuned hyperparameters ===")

xgb = XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",   # RMSE in log space = RMSLE in original space
    tree_method="hist",
    n_jobs=-1,
    early_stopping_rounds=100,
    **best_params,
)

xgb.fit(
    Xtr, y_train_log,
    eval_set=[(Xdv, y_dev_log)],
    verbose=False,
)

y_dev_log_pred = xgb.predict(Xdv)
rmsle = np.sqrt(np.mean((y_dev_log_pred - y_dev_log) ** 2))
print(f"Fixed tuned params -> dev RMSLE = {rmsle:.5f}\n")

# Retrain on full train.csv (with outliers removed) and build submission

full_train = pd.read_csv(full_train_path)
test       = pd.read_csv(test_path)

y_full = full_train["SalePrice"].values
X_full_df = full_train.drop(columns=["Id", "SalePrice"])
X_test_df = test.drop(columns=["Id"])

# Outlier removal on full train
q_gr_full = X_full_df["GrLivArea"].quantile(0.99)
q_sp_full = pd.Series(y_full).quantile(0.99)

mask_inliers_full = (X_full_df["GrLivArea"] <= q_gr_full) & (y_full <= q_sp_full)
X_full_inliers = X_full_df[mask_inliers_full].copy()
y_full_inliers = y_full[mask_inliers_full].copy()

print(f"Removed {len(y_full) - len(y_full_inliers)} outliers from full train")

# Categorical / numeric from full inliers
cat_cols_full = [c for c in X_full_inliers.columns if X_full_inliers[c].dtype == "object"]
if "MSSubClass" in X_full_inliers.columns and "MSSubClass" not in cat_cols_full:
    cat_cols_full.append("MSSubClass")
num_cols_full = [c for c in X_full_inliers.columns if c not in cat_cols_full]

for df_ in (X_full_inliers, X_test_df):
    df_[num_cols_full] = df_[num_cols_full].apply(pd.to_numeric, errors="coerce")

num_pipe_full = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

cat_pipe_full = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

feats_full = ColumnTransformer(
    transformers=[
        ("num", num_pipe_full, num_cols_full),
        ("cat", cat_pipe_full, cat_cols_full),
    ]
)

X_full = feats_full.fit_transform(X_full_inliers)
X_test = feats_full.transform(X_test_df)

print(f"Full-train features shape (inliers): {X_full.shape}")
print(f"Test features shape:                 {X_test.shape}")

y_full_log = np.log1p(y_full_inliers)

xgb_final = XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",
    n_jobs=-1,
    **best_params,
)

xgb_final.fit(X_full, y_full_log)

y_test_log_pred = xgb_final.predict(X_test)
y_test_pred = np.expm1(y_test_log_pred)
y_test_pred = np.maximum(y_test_pred, 0)

submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": y_test_pred,
})
submission.to_csv(out_path, index=False)
print(f"Wrote {out_path}")
