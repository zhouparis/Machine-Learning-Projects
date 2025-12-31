import pandas as pd
from sklearn.preprocessing import OneHotEncoder

csv_path = "my_train.csv"

df = pd.read_csv(csv_path)

# Separate inputs and target
X = df.drop(columns=["Id", "SalePrice"])
y = df["SalePrice"]

# Treat everything as categorical by converting to string
X_str = X.astype(str)

total_binary_features = 0

for col in X_str.columns:
    n_unique = X_str[col].nunique()
    contrib = n_unique - 1
    total_binary_features += contrib
    print(f"{col:20s} -> uniques: {n_unique}, contributes: {contrib}")

print("\nTotal number of binary features (sum over uniques-1):", total_binary_features)

encoder = OneHotEncoder(handle_unknown="ignore")
X_encoded = encoder.fit_transform(X_str)

print("Original X shape:", X_str.shape)
print("Encoded X shape: ", X_encoded.shape)  # (n_samples, n_binary_features)

n_binary_features = X_encoded.shape[1]
print("Number of binary (one-hot) features:", n_binary_features)
