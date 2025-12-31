# sparseAverage.py
import sys, time
import pandas as pd
from svector import svector
from heapq import nlargest, nsmallest

# ---------- IO ----------

def read_labeled(textfile):
    data = pd.read_csv(textfile)
    for i in range(len(data)):
        id, words, label = data.iloc[i]
        y = 1 if label=="+" else -1
        yield (y, words.split())

def read_test(textfile):
    data = pd.read_csv(textfile)
    for i in range(len(data)):
        id, words, label = data.iloc[i]
        y = 1 if label=="+" else -1
        yield (id, y, words)


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
    """Evaluate dev error using *averaged* params without materializing them."""
    inv_c = 1.0 / c
    err = 0
    for i, (y, words) in enumerate(read_labeled(devfile), 1):
        x = make_vector(words)
        # score_avg = (w - u/c)·x + (b - beta/c) = (w·x + b) - (u·x + beta)/c
        score = (w.dot(x) + b) - inv_c * (u.dot(x) + beta)
        err += (y * score) <= 0
    return err / i

def hardest_dev_examples(devfile, w, b, u, beta, c, k=5):
    """
    Uses read_from_forexp(devfile) -> (id, y, words_str)
    Finds:
      - top-k negatives the model is most confident are positive (highest scores among y=-1 mistakes)
      - top-k positives the model is most confident are negative (lowest scores among y=+1 mistakes)
    Scoring uses averaged perceptron: (w·x + b) - (u·x + beta)/c
    """
    inv_c = 1.0 / c
    neg_as_pos = []  # (score, id, sentence) for y=-1 misclassified
    pos_as_neg = []  # (score, id, sentence) for y=+1 misclassified

    for doc_id, y, words in read_test(devfile):
        tokens = str(words).split()
        x = make_vector(tokens)
        score = (w.dot(x) + b) - inv_c * (u.dot(x) + beta)  # averaged score

        # count ties as errors, consistent with your test_avg
        if y * score <= 0:
            sent = " ".join(tokens)
            if y == -1:
                neg_as_pos.append((score, doc_id, sent))
            else:  # y == +1
                pos_as_neg.append((score, doc_id, sent))

    top_neg_as_pos = nlargest(k, neg_as_pos, key=lambda t: t[0])   # highest scores
    top_pos_as_neg = nsmallest(k, pos_as_neg, key=lambda t: t[0])  # lowest scores
    return top_neg_as_pos, top_pos_as_neg

# ---------- Training ----------

def train_avg_perceptron(trainfile, devfile, epochs=10):
    """
    Averaged perceptron with separate bias and 'smart averaging'.
    Returns the *best averaged* (w_avg, b_avg) by dev error.
    """
    t0 = time.time()

    # current params
    w = svector()
    b = 0.0

    # smart-averaging accumulators
    u = svector()   # accumulates w updates weighted by time
    beta = 0.0      # accumulates b updates weighted by time
    c = 1           # global step (1-based, matches classic derivation)

    best_err = float("inf")
    best_w = None
    best_b = None

    top_neg_err = None
    top_pos_err = None
    for ep in range(1, epochs + 1):
        updates = 0
        # train pass
        for i, (y, words) in enumerate(read_labeled(trainfile), 1):
            x = make_vector(words)
            if y * (w.dot(x) + b) <= 0.0:
                updates += 1
                for f, v in x.items():
                    w[f]   += y * v
                    u[f]   += y * v * c
                b    += y
                beta += y * c
            c += 1

        dev_err = test_avg(devfile, w, b, u, beta, c)
        print(f"epoch {ep}, updates {updates / i * 100:.1f}%, dev {dev_err * 100:.2f}%")
        top_neg_as_pos, top_pos_as_neg = hardest_dev_examples(devfile, w, b, u, beta, c, k = 5)
        # if improved, snapshot *averaged* parameters for later use
        if dev_err < best_err:
            best_err = dev_err
            best_w = materialize_avg(w, u, c)
            best_b = b - beta / c

    # Also compute the final averaged model (in case you want it)
    final_w = materialize_avg(w, u, c)
    final_b = b - beta / c
    print(
        "Top 5 negatives predicted positive (highest scores):\n"
        + "\n".join(
            f"{idx:>2}. id={int(doc_id):>6}  score={score:>9.4f}  |  {text}"
            for idx, (score, doc_id, text) in enumerate(top_neg_as_pos, 1)
        )
        + "\n\n"
        + "Top 5 positives predicted negative (lowest scores):\n"
        + "\n".join(
            f"{idx:>2}. id={int(doc_id):>6}  score={score:>9.4f}  |  {text}"
            for idx, (score, doc_id, text) in enumerate(top_pos_as_neg, 1)
        )
    )
    print(f"best dev err {best_err * 100:.2f}%, |w|={len(best_w)}, time: {time.time()-t0:.1f}s")
    return best_w, best_b, final_w  # return the best averaged params

# ---------- Inference ----------

def predict(x, w, b):
    return 1 if (w.dot(x) + b) >= 0 else -1

def predict_to_csv(testfile, w, b, out_csv="submission.csv"):
    rows = []
    for id, label, words in read_test(testfile):
        x = make_vector(words)
        score = w.dot(x) + b
        yhat = 1 if (score) > 0 else -1
        pred = "+" if yhat == 1 else "-"
        rows.append((id, words, pred))
    pd.DataFrame(rows, columns=["id", "sentence", "target"]).to_csv(out_csv, index=False)


# ---------- CLI ----------

if __name__ == "__main__":
    trainfile, devfile, testfile, out_csv = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    w, b, final_w = train_avg_perceptron(trainfile, devfile, epochs=10)
    predict_to_csv(testfile, w, b, out_csv=out_csv)

    pos20, neg20 = top_features(final_w, n=20, eps=1e-12)
    print("Top 20 positive:")
    for f, wt in pos20:
        print(f"{f}\t{wt:.4f}")

    print("\nTop 20 negative:")
    for f, wt in neg20:
        print(f"{f}\t{wt:.4f}")
