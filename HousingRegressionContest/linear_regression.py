import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# ----  Load data ----
train_path = "my_train.csv"
dev_path   = "my_dev.csv"

train_df = pd.read_csv(train_path)
dev_df   = pd.read_csv(dev_path)

# ----  Split into X (features) and y (target) ----
X_train = train_df.drop(columns=["Id", "SalePrice"])
y_train = train_df["SalePrice"].values

X_dev   = dev_df.drop(columns=["Id", "SalePrice"])
y_dev   = dev_df["SalePrice"].values

# ----  Convert all features to string and one-hot encode ----
X_train_str = X_train.astype(str)
X_dev_str   = X_dev.astype(str)

encoder = OneHotEncoder(handle_unknown="ignore",sparse_output=True)
X_train_enc = encoder.fit_transform(X_train_str)
X_dev_enc   = encoder.transform(X_dev_str)

print("Encoded train shape:", X_train_enc.shape)
print("Encoded dev shape:  ", X_dev_enc.shape)

# ----  Work in log output space: log1p(y) ----
y_train_log = np.log1p(y_train)
y_dev_log   = np.log1p(y_dev)

# ----  Train linear regression on log-prices ----
linreg = LinearRegression()
linreg.fit(X_train_enc, y_train_log)

# ----  Predict on dev and compute RMSLE (RMSE in log space) ----
y_dev_log_pred = linreg.predict(X_dev_enc)

# RMSLE = sqrt(mean((log1p(y_pred) - log1p(y_true))^2))
# Here, y_dev_log_pred ≈ log1p(y_pred) and y_dev_log is log1p(y_true)
rmsle = np.sqrt(np.mean((y_dev_log_pred - y_dev_log) ** 2))
print("RMSLE on dev:", rmsle)


