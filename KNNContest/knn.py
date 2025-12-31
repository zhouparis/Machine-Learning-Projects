import time
from typing import Literal
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Load
train = pd.read_csv("hw1-data/income.train.5k.csv", header=0, skipinitialspace=True, na_filter=False)
dev   = pd.read_csv("hw1-data/income.dev.csv",      header=0, skipinitialspace=True, na_filter=False)

# Columns
num_cols = ["age", "hours"]
cat_cols = ["sector", "edu", "marriage", "occupation", "race", "sex", "country"]
feature_cols = num_cols + cat_cols

Xtr_df = train[feature_cols].copy()
Xdv_df = dev[feature_cols].copy()
y_train = (train["target"] == ">50K").astype(int).values
y_dev   = (dev["target"]   == ">50K").astype(int).values

# Preprocessor: scale numerics to [0,2]; binarize categoricals
pre = ColumnTransformer(
    transformers=[
        ("num", MinMaxScaler(feature_range=(0, 2)), num_cols),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop",
)
X_train = pre.fit_transform(Xtr_df)      # fit on train only
X_dev   = pre.transform(Xdv_df)

try:
    feat_names = pre.get_feature_names_out()
    print("Feature dim:", len(feat_names))
except Exception:
    print("Feature dim:", X_train.shape[1])


# Query: first dev row
q = X_dev[0:1]   # shape (1, d)

# --- Euclidean (L2) with sklearn
nbrs_L2 = NearestNeighbors(n_neighbors=3, metric="euclidean", n_jobs=-1).fit(X_train)
dist_L2, idx_L2 = nbrs_L2.kneighbors(q, return_distance=True)
print("Top-3 (Euclidean) indices:", idx_L2[0].tolist())
print("Top-3 (Euclidean) dists:  ", [float(f"{x:.3f}") for x in dist_L2[0]])

# --- Manhattan (L1) with sklearn
nbrs_L1 = NearestNeighbors(n_neighbors=3, metric="manhattan", n_jobs=-1).fit(X_train)
dist_L1, idx_L1 = nbrs_L1.kneighbors(q, return_distance=True)
print("Top-3 (Manhattan) indices:", idx_L1[0].tolist())
print("Top-3 (Manhattan) dists:  ", [float(f"{x:.3f}") for x in dist_L1[0]])

# --- Manual distances to those same 3 indices (both norms)
def dists_to_indices(X, p, indices, p_norm=2):
    V = X[indices] - p  # broadcast: (k,d)
    if p_norm == 2:
        return np.linalg.norm(V, axis=1)
    elif p_norm == 1:
        return np.abs(V).sum(axis=1)
    else:
        raise ValueError("p_norm must be 1 or 2")

print("\nManual check against sklearn's 3 Euclidean neighbors:")
print("Euclidean:", [float(f"{x:.3f}") for x in dists_to_indices(X_train, q, idx_L2[0], p_norm=2)])
print("Manhattan:", [float(f"{x:.3f}") for x in dists_to_indices(X_train, q, idx_L2[0], p_norm=1)])

# Optional: assert L2 distances match up to 1e-9
np.testing.assert_allclose(dist_L2[0], dists_to_indices(X_train, q, idx_L2[0], 2), rtol=1e-7, atol=1e-9)

def knn_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: np.ndarray,
    k: int,
    metric: Literal["euclidean","manhattan"]="euclidean",
) -> np.ndarray:
    """
    Returns predictions for X_query using uniform-weight k-NN on X_train/y_train.
    Tie-break: smaller class index (0 over 1), to mimic sklearn's behavior.
    """
    # pairwise distances between all queries and all training points
    # For memory friendliness, do in batches if needed; here sizes are small enough to do at once.
    if metric == "euclidean":
        # ||x - q||_2  = sqrt(sum((x - q)^2, axis=1))
        # Compute in blocks: D[i] = distances for query i
        D = np.empty((X_query.shape[0], X_train.shape[0]), dtype=np.float32)
        for i in range(X_query.shape[0]):
            diff = X_train - X_query[i]              # (n_train, d)
            D[i] = np.sqrt(np.einsum("ij,ij->i", diff, diff, optimize=True))
    elif metric == "manhattan":
        D = np.empty((X_query.shape[0], X_train.shape[0]), dtype=np.float32)
        for i in range(X_query.shape[0]):
            D[i] = np.abs(X_train - X_query[i]).sum(axis=1)
    else:
        raise ValueError("metric must be 'euclidean' or 'manhattan'")

    # get indices of k smallest distances without full sorting
    # argpartition yields k unordered nearest; then we can sort just those k if we care (optional)
    kth = np.argpartition(D, kth=k-1, axis=1)[:, :k]  # (n_query, k)

    # majority vote (binary labels 0/1)
    # y_train[kth] has shape (n_query, k); sum across axis=1 gives # of ones
    ones = y_train[kth].sum(axis=1)
    zeros = k - ones
    # Predict 1 if more ones than zeros; tie -> 0
    y_pred = (ones > zeros).astype(int)
    return y_pred

# Convenience: evaluate error% and +rate
def err_and_pos(y_true, y_pred):
    err = 100.0 * (1.0 - (y_true == y_pred).mean())
    pos = 100.0 * (y_pred == 1).mean()
    return err, pos

# Sweep k = 1..99 (odd), Euclidean
rows_L2 = []
t0 = time.perf_counter()
for k in range(1, 100, 2):
    yhat_tr = knn_predict(X_train, y_train, X_train, k, metric="euclidean")
    yhat_dv = knn_predict(X_train, y_train, X_dev,   k, metric="euclidean")
    tr_err, tr_pos = err_and_pos(y_train, yhat_tr)
    dv_err, dv_pos = err_and_pos(y_dev,   yhat_dv)
    rows_L2.append((k, tr_err, tr_pos, dv_err, dv_pos))
t1 = time.perf_counter()

for k, tr_err, tr_pos, dv_err, dv_pos in rows_L2:
    print(f"k={k:2d} train_err {tr_err:5.1f}% (+:{tr_pos:5.1f}%)  dev_err {dv_err:5.1f}% (+:{dv_pos:5.1f}%)")

best_L2 = min(rows_L2, key=lambda r: (r[3], r[0]))
print(f"\n[Euclidean] Best dev error: {best_L2[3]:.2f}% at k={best_L2[0]}  (+rate {best_L2[4]:.1f}%)")
print(f"Time to sweep (your kNN, Euclidean): {t1 - t0:.2f}s")

# Sweep k = 1..99 (odd), Manhattan
rows_L1 = []
t0 = time.perf_counter()
for k in range(1, 100, 2):
    yhat_tr = knn_predict(X_train, y_train, X_train, k, metric="manhattan")
    yhat_dv = knn_predict(X_train, y_train, X_dev,   k, metric="manhattan")
    tr_err, tr_pos = err_and_pos(y_train, yhat_tr)
    dv_err, dv_pos = err_and_pos(y_dev,   yhat_dv)
    rows_L1.append((k, tr_err, tr_pos, dv_err, dv_pos))
t1 = time.perf_counter()

for k, tr_err, tr_pos, dv_err, dv_pos in rows_L1:
    print(f"[L1] k={k:2d} train_err {tr_err:5.1f}% (+:{tr_pos:5.1f}%)  dev_err {dv_err:5.1f}% (+:{dv_pos:5.1f}%)")

best_L1 = min(rows_L1, key=lambda r: (r[3], r[0]))
print(f"\n[Manhattan] Best dev error: {best_L1[3]:.2f}% at k={best_L1[0]}  (+rate {best_L1[4]:.1f}%)")
print(f"Time to sweep (your kNN, Manhattan): {t1 - t0:.2f}s")
