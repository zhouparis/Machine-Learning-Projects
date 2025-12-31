#!/usr/bin/env python3
import sys, time, argparse
import pandas as pd
from collections import Counter
from svector import svector

# ---------- IO ----------

def read_labeled(textfile):
    """Yield (y, tokens) from a CSV with columns [id, sentence|words, label] (in that order)."""
    data = pd.read_csv(textfile)
    for i in range(len(data)):
        _id, text, label = data.iloc[i]
        y = 1 if label == "+" else -1
        yield y, str(text).split()

def read_test(textfile):
    """Yield (id, raw_sentence_string) from a CSV without labels."""
    df = pd.read_csv(textfile)
    id_col = "id" if "id" in df.columns else df.columns[0]
    if "sentence" in df.columns:
        txt_col = "sentence"
    elif "words" in df.columns:
        txt_col = "words"
    else:
        txt_col = df.columns[1]
    for i in range(len(df)):
        _id = df.at[i, id_col]
        text = df.at[i, txt_col]
        yield _id, str(text)

# ---------- Vocab & vectors ----------

def count_vocab(trainfile):
    """Return Counter of token frequencies from the training CSV."""
    cnt = Counter()
    for _, tokens in read_labeled(trainfile):
        cnt.update(tokens)
    return cnt

def build_keep(counter, min_count):
    """Return a set of tokens whose freq >= min_count."""
    if min_count <= 1:
        return None  # None = keep everything
    return {w for w, c in counter.items() if c >= min_count}

def make_vector(words, keep=None):
    v = svector()
    for w in words:
        if (keep is None) or (w in keep):
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

def test_avg(devfile, w, b, u, beta, c, keep=None):
    """Evaluate dev error using averaged params without materializing them."""
    inv_c = 1.0 / c
    err = 0
    n = 0
    for n, (y, words) in enumerate(read_labeled(devfile), 1):
        x = make_vector(words, keep)
        # score_avg = (w·x + b) - (u·x + beta)/c
        score = (w.dot(x) + b) - inv_c * (u.dot(x) + beta)
        err += (y * score) <= 0
    return (err / n) if n else 0.0

# ---------- Training (Averaged Perceptron) ----------

def train_avg_perceptron(trainfile, devfile, epochs=10, min_count=1):
    """
    Averaged perceptron with explicit bias and 'smart averaging'.
    Returns the *best averaged* (w_avg, b_avg, best_err, keep_set).
    """
    t0 = time.time()

    # build pruned vocab
    vocab_t0 = time.time()
    counter = count_vocab(trainfile)
    keep = build_keep(counter, min_count)
    kept = len(counter) if keep is None else len(keep)
    pruned = 0 if keep is None else (len(counter) - len(keep))
    print(f"[vocab] total={len(counter)} kept={kept} pruned={pruned} (min_count={min_count}) in {time.time()-vocab_t0:.2f}s")

    # current params
    w = svector()
    b = 0.0

    # smart-averaging accumulators
    u = svector()
    beta = 0.0
    c = 1  # global step

    best_err = float("inf")
    best_w_avg = None
    best_b_avg = None

    for ep in range(1, epochs + 1):
        ep_t0 = time.time()
        updates = 0
        seen = 0
        for seen, (y, words) in enumerate(read_labeled(trainfile), 1):
            x = make_vector(words, keep)
            if y * (w.dot(x) + b) <= 0.0:  # mistake or tie
                updates += 1
                for f, v in x.items():
                    w[f]   += y * v
                    u[f]   += y * v * c
                b    += y
                beta += y * c
            c += 1

        dev_err = test_avg(devfile, w, b, u, beta, c, keep)
        print(f"epoch {ep}, updates {updates / seen * 100:.1f}%, dev {dev_err * 100:.2f}%, time {time.time()-ep_t0:.2f}s")

        if dev_err < best_err:
            best_err = dev_err
            best_w_avg = materialize_avg(w, u, c)
            best_b_avg = b - beta / c

    print(f"[done] best dev err {best_err * 100:.2f}%, |w|={len(best_w_avg)}, total time {time.time()-t0:.1f}s")
    return best_w_avg, best_b_avg, best_err, keep

# ---------- Inference / Kaggle Export ----------

def predict_df(testfile, model, b, keep=None):
    """Return DataFrame ['id','sentence','prediction'] using the same pruned vocab."""
    rows = []
    for _id, raw in read_test(testfile):
        x = make_vector(raw.split(), keep)
        score = model.dot(x) + b
        yhat = 1 if score > 0 else -1   # ties -> -1
        pred = "+" if yhat == 1 else "-"
        rows.append((_id, raw, pred))
    return pd.DataFrame(rows, columns=["id", "sentence", "prediction"])

def predict_to_csv(testfile, model, b, out_csv="submission.csv", keep=None):
    df = predict_df(testfile, model, b, keep)
    df.to_csv(out_csv, index=False)
    return df

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_csv")
    ap.add_argument("dev_csv")
    ap.add_argument("test_csv")
    ap.add_argument("out_csv")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--min_count", type=int, default=1, help="keep tokens with freq >= min_count")
    args = ap.parse_args()

    w_best, b_best, best_err, keep = train_avg_perceptron(
        args.train_csv, args.dev_csv, epochs=args.epochs, min_count=args.min_count
    )
    df_sub = predict_to_csv(args.test_csv, w_best, b_best, args.out_csv, keep)
    print(f"Wrote {len(df_sub)} rows to {args.out_csv}")

if __name__ == "__main__":
    main()
