# xgb_bow.py
import pandas as pd
import numpy as np
from collections import Counter
from scipy import sparse as sp
from xgboost.callback import EarlyStopping
from xgboost import XGBClassifier

# -------------------- IO --------------------

def read_labeled(path):
    """
    Expect columns: id, words, label (label is '+' or '-').
    Yields (id:int, y:int in {0,1}, tokens:list[str])
    """
    df = pd.read_csv(path, keep_default_na=False)
    need = {"id", "sentence", "target"}
    if not need.issubset(df.columns):
        raise ValueError(f"Expected columns {need}, got {set(df.columns)}")
    for _id, words, lab in df[["id", "sentence", "target"]].itertuples(index=False, name=None):
        y = 1 if str(lab).strip() == "+" else 0
        yield int(_id), y, str(words).split()

def read_unlabeled(path):
    """
    Expect columns: id, words.
    Yields (id:int, tokens:list[str])
    """
    df = pd.read_csv(path, keep_default_na=False)
    need = {"id", "sentence"}
    if not need.issubset(df.columns):
        raise ValueError(f"Expected columns {need}, got {set(df.columns)}")
    for _id, words in df[["id", "sentence"]].itertuples(index=False, name=None):
        yield int(_id), str(words).split()

# -------------------- Vocab & Vectorization --------------------

def build_vocab(trainfile, min_count=2, max_features=None):
    """
    Count tokens on training set and keep those with freq >= min_count.
    Optionally cap at max_features most frequent.
    Returns dict token->index
    """
    cnt = Counter()
    for _id, y, toks in read_labeled(trainfile):
        cnt.update(toks)
    # prune
    items = [(tok, c) for tok, c in cnt.items() if c >= min_count]
    # keep most frequent if capped
    if max_features is not None and len(items) > max_features:
        items = sorted(items, key=lambda kv: kv[1], reverse=True)[:max_features]
    # index tokens
    vocab = {tok: i for i, (tok, _) in enumerate(sorted(items))}
    return vocab

def rows_to_csr(examples, vocab):
    """
    Convert iterable of tokenized docs into CSR matrix (count BOW).
    examples: iterable of tokens (list[str])
    Returns X (csr), indptr aligned with order of examples.
    """
    data, indices, indptr = [], [], [0]
    for toks in examples:
        local = {}
        for w in toks:
            j = vocab.get(w)
            if j is not None:
                local[j] = local.get(j, 0.0) + 1.0
        if local:
            idxs, vals = zip(*sorted(local.items()))
            indices.extend(idxs)
            data.extend(vals)
        indptr.append(len(indices))
    X = sp.csr_matrix((np.array(data, dtype=np.float32),
                       np.array(indices, dtype=np.int32),
                       np.array(indptr, dtype=np.int32)),
                      shape=(len(indptr)-1, len(vocab)),
                      dtype=np.float32)
    return X

def vectorize_labeled(path, vocab):
    ids, ys, docs = [], [], []
    for _id, y, toks in read_labeled(path):
        ids.append(_id); ys.append(y); docs.append(toks)
    X = rows_to_csr(docs, vocab)
    y = np.array(ys, dtype=np.int32)
    return ids, X, y

def vectorize_unlabeled(path, vocab):
    ids, docs = [], []
    for _id, toks in read_unlabeled(path):
        ids.append(_id); docs.append(toks)
    X = rows_to_csr(docs, vocab)
    return ids, X

# -------------------- Training + Inference --------------------

def train_xgb(trainfile, devfile, testfile,
              out_csv="submission_xgb.csv",
              sep=",",
              min_count=2,           # prune singletons by default
              max_features=None,     # optionally cap vocab size
              random_state=42):
    # 1) Vocab
    vocab = build_vocab(trainfile, min_count=min_count,
                        max_features=max_features)
    print(f"Vocab size after pruning: {len(vocab)}")

    # 2) Vectorize
    train_ids, Xtr, ytr = vectorize_labeled(trainfile, vocab)
    dev_ids,   Xdv, ydv = vectorize_labeled(devfile,   vocab)
    test_ids,  Xte      = vectorize_unlabeled(testfile, vocab)

    best_acc = -1

    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="error",
        tree_method="hist",
        n_estimators=10000,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        n_jobs=-1,
        early_stopping_rounds=200,
        random_state=42,
        learning_rate=0.03,
        max_depth=8,
        min_child_weight=1,
    )
    clf.fit(Xtr, ytr, eval_set=[(Xdv, ydv)], verbose=False)
    acc = (clf.predict(Xdv) == ydv).mean()
    if acc > best_acc:
        best_acc, best_model = acc, clf

    print("Best dev acc:", best_acc)

    # 5) Dev accuracy (quick sanity check)
    dev_pred = (clf.predict_proba(Xdv)[:, 1] >= 0.5).astype(int)
    dev_acc = (dev_pred == ydv).mean()
    print(f"Dev accuracy: {dev_acc*100:.2f}%")

    # 6) Predict test and write CSV (id,target with + / -)
    proba = clf.predict_proba(Xte)[:, 1]
    yhat = (proba >= 0.5).astype(int)
    label_map = {1: "+", 0: "-"}
    rows = [(i, label_map[int(y)]) for i, y in zip(test_ids, yhat)]
    rows.sort(key=lambda r: r[0])
    pd.DataFrame(rows, columns=["id", "target"]).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(rows)} rows.")

# -------------------- Example CLI --------------------
if __name__ == "__main__":
    import sys
    # Usage: python xgb_bow.py train.csv dev.csv test.csv submission_xgb.csv
    trainfile, devfile, testfile, out_csv = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    train_xgb(trainfile, devfile, testfile, out_csv=out_csv,
              min_count=2, max_features=None)

