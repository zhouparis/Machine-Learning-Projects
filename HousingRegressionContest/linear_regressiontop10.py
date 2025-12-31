import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# ---------- Load training data ----------
train_path = "my_train.csv"   # change if needed
df = pd.read_csv(train_path)

# ---------- Separate inputs and target ----------
X = df.drop(columns=["Id", "SalePrice"])
y = df["SalePrice"].values

# Convert all features to string to treat everything as categorical
X_str = X.astype(str)

# ---------- One-hot encode features ----------
encoder = OneHotEncoder(handle_unknown="ignore")
X_enc = encoder.fit_transform(X_str)

print("Encoded X shape:", X_enc.shape)

# ---------- Work in log output space ----------
y_log = np.log1p(y)  # log(1 + SalePrice)

# ---------- Train linear regression on log-prices ----------
linreg = LinearRegression()
linreg.fit(X_enc, y_log)

# ---------- Get feature names and coefficients ----------
feature_names = encoder.get_feature_names_out(X_str.columns)
coefs = linreg.coef_ 

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coef": coefs
})

# ---------- Top 10 positive and negative features ----------
top_positive = coef_df.sort_values("coef", ascending=False).head(10)
top_negative = coef_df.sort_values("coef", ascending=True).head(10)

print("\nTop 10 POSITIVE features (increase log(SalePrice)):")
print(top_positive.to_string(index=False))

print("\nTop 10 NEGATIVE features (decrease log(SalePrice)):")
print(top_negative.to_string(index=False))


b = linreg.intercept_
typical_price = np.expm1(b)

print(typical_price)