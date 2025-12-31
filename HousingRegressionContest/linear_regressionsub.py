import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor

# ---------- Load data ----------
train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")

# ---------- Split X / y ----------
X_train = train_df.drop(columns=["Id", "SalePrice"])
y_train = train_df["SalePrice"].values

test_ids = test_df["Id"].values
X_test   = test_df.drop(columns=["Id"])

# ---------- Treat all features as categorical (string) and one-hot ----------

def to_categorical_str(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    out = df.copy()
    out = out.astype("string")
    out = out.fillna("")
    out = out.replace("NA", "")
    return out

X_train_str = to_categorical_str(X_train)
X_test_str  = to_categorical_str(X_test)

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=True
)

X_train_enc = encoder.fit_transform(X_train_str)
X_test_enc  = encoder.transform(X_test_str)

print("Encoded train shape:", X_train_enc.shape)
print("Encoded test shape: ", X_test_enc.shape)

# ---------- Linear regression with non-linear link on target ----------

reg = TransformedTargetRegressor(
    regressor=LinearRegression(),
    func=np.log1p,
    inverse_func=np.expm1,
)

reg.fit(X_train_enc, y_train)

# ---------- Predict on test (already in price space) ----------
y_test_pred = reg.predict(X_test_enc)

# ---------- Build submission ----------
submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": y_test_pred
})

submission_path = "my_linreg_submission_ttr.csv"
submission.to_csv(submission_path, index=False)

print("Wrote submission to:", submission_path)
print("Pred min / max:", y_test_pred.min(), y_test_pred.max())
print(submission.head())
