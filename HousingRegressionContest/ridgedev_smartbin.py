import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_log_error

# Tune alpha on my_train / my_dev

train_df = pd.read_csv("my_train.csv")
dev_df   = pd.read_csv("my_dev.csv")

X_train = train_df.drop(columns=["Id", "SalePrice"])
y_train = train_df["SalePrice"].values

X_dev = dev_df.drop(columns=["Id", "SalePrice"])
y_dev = dev_df["SalePrice"].values

# Categorical columns from training
cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]
if "MSSubClass" in X_train.columns and "MSSubClass" not in cat_cols:
    cat_cols.append("MSSubClass")

num_cols = [c for c in X_train.columns if c not in cat_cols]

# Ensure numeric columns are numeric
X_train[num_cols] = X_train[num_cols].apply(pd.to_numeric, errors="coerce")
X_dev[num_cols]   = X_dev[num_cols].apply(pd.to_numeric, errors="coerce")

num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols),
    ]
)

alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
best_alpha = None
best_rmsle = float("inf")

for alpha in alphas:
    reg = TransformedTargetRegressor(
        regressor=Ridge(alpha=alpha, random_state=42),
        func=np.log1p,
        inverse_func=np.expm1,
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", reg),
    ])

    model.fit(X_train, y_train)
    y_dev_pred = model.predict(X_dev)
    rmsle = np.sqrt(mean_squared_log_error(y_dev, y_dev_pred))

    print(f"alpha={alpha:.2g} -> dev RMSLE = {rmsle:.5f}")

    if rmsle < best_rmsle:
        best_rmsle = rmsle
        best_alpha = alpha

print(f"\nBest alpha on dev: {best_alpha} with RMSLE = {best_rmsle:.5f}")

# Retrain on full train.csv with best_alpha and build Kaggle submission

full_train = pd.read_csv("train.csv")
test_df    = pd.read_csv("test.csv")

X_full = full_train.drop(columns=["Id", "SalePrice"])
y_full = full_train["SalePrice"].values

X_test = test_df.drop(columns=["Id"])
test_ids = test_df["Id"]

# Recompute cat/num cols based on full train (same logic)
cat_cols_full = [c for c in X_full.columns if X_full[c].dtype == "object"]
if "MSSubClass" in X_full.columns and "MSSubClass" not in cat_cols_full:
    cat_cols_full.append("MSSubClass")

num_cols_full = [c for c in X_full.columns if c not in cat_cols_full]

X_full[num_cols_full] = X_full[num_cols_full].apply(pd.to_numeric, errors="coerce")
X_test[num_cols_full] = X_test[num_cols_full].apply(pd.to_numeric, errors="coerce")

num_transformer_full = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

cat_transformer_full = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor_full = ColumnTransformer(
    transformers=[
        ("num", num_transformer_full, num_cols_full),
        ("cat", cat_transformer_full, cat_cols_full),
    ]
)

reg_full = TransformedTargetRegressor(
    regressor=Ridge(alpha=best_alpha, random_state=42),
    func=np.log1p,
    inverse_func=np.expm1,
)

final_model = Pipeline(steps=[
    ("preprocessor", preprocessor_full),
    ("regressor", reg_full),
])

final_model.fit(X_full, y_full)
y_test_pred = final_model.predict(X_test)
y_test_pred = np.maximum(y_test_pred, 0)

submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": y_test_pred,
})

submission_path = "ridge_smart_binarization_submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote submission to:", submission_path)
