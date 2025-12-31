import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error

# ---------- Load train / dev ----------
train_df = pd.read_csv("my_train.csv")
dev_df   = pd.read_csv("my_dev.csv")

# Separate X, y
X_train = train_df.drop(columns=["Id", "SalePrice"])
y_train = train_df["SalePrice"].values

X_dev = dev_df.drop(columns=["Id", "SalePrice"])
y_dev = dev_df["SalePrice"].values

# ---------- Define categorical and numeric columns ----------
# Categorical columns are those with object dtype in *training* data
cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]

# Treat MSSubClass as categorical even though it's int
if "MSSubClass" in X_train.columns and "MSSubClass" not in cat_cols:
    cat_cols.append("MSSubClass")

# Everything else is numeric
num_cols = [c for c in X_train.columns if c not in cat_cols]

# Force numeric columns to be numeric in BOTH train and dev
X_train[num_cols] = X_train[num_cols].apply(pd.to_numeric, errors="coerce")
X_dev[num_cols]   = X_dev[num_cols].apply(pd.to_numeric, errors="coerce")

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
    func=np.log1p,
    inverse_func=np.expm1,
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", reg),
])

# ---------- Train and evaluate ----------
model.fit(X_train, y_train)
y_dev_pred = model.predict(X_dev)

rmsle = np.sqrt(mean_squared_log_error(y_dev, y_dev_pred))
print("RMSLE on dev:", rmsle)

# ---------- Total number of features ----------
pre = model.named_steps["preprocessor"]           # fitted ColumnTransformer
X_trans = pre.transform(X_train)                  # transform training X to get shape
n_features = X_trans.shape[1]
print("Total number of features:", n_features)

# ---------- Top 10 positive / negative features ----------
feature_names = pre.get_feature_names_out()       # names after all transforms

lin = model.named_steps["regressor"].regressor_   # underlying LinearRegression
coefs = np.ravel(lin.coef_)                       # ensure 1D

print("len(feature_names) =", len(feature_names))
print("len(coefs)         =", len(coefs))

coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})

top_pos = coef_df.sort_values("coef", ascending=False).head(10)
top_neg = coef_df.sort_values("coef", ascending=True).head(10)

print("\nTop 10 POSITIVE:")
print(top_pos.to_string(index=False))

print("\nTop 10 NEGATIVE:")
print(top_neg.to_string(index=False))
