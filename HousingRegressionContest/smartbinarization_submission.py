import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

# ---------- Load full train and test ----------
train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")

# Separate X, y for training
X_train = train_df.drop(columns=["Id", "SalePrice"])
y_train = train_df["SalePrice"].values

# Test features (keep Id for submission)
X_test = test_df.drop(columns=["Id"])
test_ids = test_df["Id"]

# ---------- Define categorical and numeric columns ----------
# Categorical columns: object dtype
cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]

# Treat MSSubClass as categorical even though it's int
if "MSSubClass" in X_train.columns and "MSSubClass" not in cat_cols:
    cat_cols.append("MSSubClass")

# Everything else is numeric
num_cols = [c for c in X_train.columns if c not in cat_cols]

# Ensure numeric columns are actually numeric in BOTH train and test
X_train[num_cols] = X_train[num_cols].apply(pd.to_numeric, errors="coerce")
X_test[num_cols]  = X_test[num_cols].apply(pd.to_numeric, errors="coerce")

# ---------- Preprocessing ----------
# Numeric: impute median + scale
num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# Categorical: impute most frequent, then one-hot encode
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

# ---------- Log-link linear regression ----------
reg = TransformedTargetRegressor(
    regressor=LinearRegression(),
    func=np.log1p,    # fit on log1p(price)
    inverse_func=np.expm1,  # predict back in price space
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", reg),
])

# ---------- Train on full train.csv ----------
model.fit(X_train, y_train)

# ---------- Predict on test.csv ----------
y_test_pred = model.predict(X_test)  # already in dollars
y_test_pred = np.maximum(y_test_pred, 0)  # safety clip, just in case

# ---------- Build Kaggle submission ----------
submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": y_test_pred,
})

submission_path = "smart_binarization_submission.csv"
submission.to_csv(submission_path, index=False)

print("Wrote submission to:", submission_path)
