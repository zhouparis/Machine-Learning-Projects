import sys, time
import pandas as pd
from svector import svector

# ---------- IO ----------

def read_labeled(textfile):
    """Yield (y, tokens) from a CSV with columns [id, sentence|words, label]."""
    data = pd.read_csv(textfile)
    # assume columns are in order [id, text, label]
    for i in range(len(data)):
        _id, text, label = data.iloc[i]
        y = 1 if label == "+" else -1
        yield y, text.split()

def read_test(textfile):
    """
    Yield (id, raw_sentence_string) from a CSV without labels.
    Accepts text column named 'sentence' or 'words' (case-sensitive).
    Falls back to 2nd column if names are unknown.
    """
    df = pd.read_csv(textfile)
    # pick id column
    id_col = "id" if "id" in df.columns else df.columns[0]
    # pick text column
    if "sentence" in df.columns:
        txt_col = "sentence"
    elif "words" in df.columns:
        txt_col = "words"
    else:
        # fallback: assume second column is the text
        txt_col = df.columns[1]
    for i in range(len(df)):
        _id = df.at[i, id_col]
        text = df.at[i, txt_col]
        yield _id, text

# ---------- Sparse helpers ----------

def make_vector(words):
    v = svector()
    for w in words:
        v[w] += 1.0
    return v

def materialize_avg(w, u, c):
    """Return a new svector equal to w - u/c (sparse)."""
    inv_c = 1.0 / c
    out = svector()
    for f, v in w.items():
        val = v - u[f] * inv_c
        if val != 0.0:
            out[f] = val
    return out

def test_avg(devfile, w, b, u, beta, c):
    """Evaluate dev error using averaged params without materializing them."""
    inv_c = 1.0 / c
    err = 0
    n = 0
    for n, (y, words) in enumerate(read_labeled(devfile), 1):
        x = make_vector(words)
        # score_avg = (w - u/c)·x + (b - beta/c) = (w·x + b) - (u·x + beta)/c
        score = (w.dot(x) + b) - inv_c * (u.dot(x) + beta)
        err += (y * score) <= 0
    return (err / n) if n else 0.0

# ---------- Training (Averaged Perceptron) ----------

def train_avg_perceptron(trainfile, devfile, epochs=10):
    """
    Averaged perceptron with separate bias and 'smart averaging'.
    Returns the *best averaged* (w_avg, b_avg, best_err).
    """
    t0 = time.time()

    # current params
    w = svector()
    b = 0.0

    # smart-averaging accumulators
    u = svector()   # accumulates w updates weighted by time
    beta = 0.0      # accumulates b updates weighted by time
    c = 1           # global step (1-based)

    best_err = float("inf")
    best_w_avg = None
    best_b_avg = None

    for ep in range(1, epochs + 1):
        updates = 0
        for i, (y, words) in enumerate(read_labeled(trainfile), 1):
            x = make_vector(words)
            if y * (w.dot(x) + b) <= 0.0:  # mistake or tie
                updates += 1
                for f, v in x.items():
                    w[f]   += y * v
                    u[f]   += y * v * c
                b    += y
                beta += y * c
            c += 1

        dev_err = test_avg(devfile, w, b, u, beta, c)
        print(f"epoch {ep}, updates {updates / i * 100:.1f}%, dev {dev_err * 100:.2f}%")

        if dev_err < best_err:
            best_err = dev_err
            best_w_avg = materialize_avg(w, u, c)
            best_b_avg = b - beta / c

    print(f"best dev err {best_err * 100:.2f}%, |w|={len(best_w_avg)}, time: {time.time()-t0:.1f}s")
    return best_w_avg, best_b_avg, best_err

# ---------- Inference / Kaggle Export ----------

def predict_df(testfile, model, b):
    """
    Return a DataFrame with columns ['id','sentence','prediction'].
    - 'sentence' is the original raw string from the test CSV.
    - 'prediction' is '+' or '-'.
    """
    rows = []
    for _id, raw in read_test(testfile):
        x = make_vector(raw.split())
        score = model.dot(x) + b
        yhat = 1 if score > 0 else -1   # ties -> -1, consistent with <= 0 error rule
        pred = "+" if yhat == 1 else "-"
        rows.append((_id, raw, pred))
    return pd.DataFrame(rows, columns=["id", "sentence", "prediction"])

def predict_to_csv(testfile, model, b, out_csv="submission.csv"):
    df = predict_df(testfile, model, b)
    df.to_csv(out_csv, index=False)
    return df

# ---------- CLI ----------

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python best_model.py <train.csv> <dev.csv> <test.csv> <submission.csv>")
        sys.exit(1)
    trainfile, devfile, testfile, out_csv = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    w_best, b_best, _ = train_avg_perceptron(trainfile, devfile, epochs=10)
    df_sub = predict_to_csv(testfile, w_best, b_best, out_csv)
    print(f"Wrote {len(df_sub)} rows to {out_csv}")