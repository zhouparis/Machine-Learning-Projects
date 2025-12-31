import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

if len(sys.argv) != 5:
    print("Usage: python xgbTrain.py <train.csv> <dev.csv> <test.csv> <submission.csv>")
    sys.exit(1)

train_path, dev_path, test_path, out_path = sys.argv[1:5]

# --- Load ---
train = pd.read_csv(train_path)
dev   = pd.read_csv(dev_path)
test  = pd.read_csv(test_path)

text_col = "sentence" if "sentence" in train.columns else "words"
label_col = "label" if "label" in train.columns else "target" 

Xtr_text = train[text_col].astype(str).values
Xdv_text = dev[text_col].astype(str).values
Xte_text = test[text_col].astype(str).values

y_train = (train[label_col] == "+").astype(int).values
y_dev   = (dev[label_col] == "+").astype(int).values

# --- TF-IDF (word + char) ---
tfidf_word = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=2, max_df=0.95, sublinear_tf=True)
tfidf_char = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2, max_df=0.95, sublinear_tf=True)
feats = FeatureUnion([("w", tfidf_word), ("c", tfidf_char)])

Xtr = feats.fit_transform(Xtr_text)
Xdv = feats.transform(Xdv_text)
Xte = feats.transform(Xte_text)

# Class imbalance
pos = y_train.sum(); neg = len(y_train) - pos
scale_pos_weight = (neg / pos) if pos and neg else 1.0

xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    learning_rate=0.08,
    max_depth=6,
    min_child_weight=2,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.0,
    reg_lambda=1.0,
    reg_alpha=0.0,
    n_estimators=3000,
    tree_method="hist",
    n_jobs=-1,
    early_stopping_rounds=150,
    scale_pos_weight=scale_pos_weight,
)

xgb.fit(
    Xtr, y_train,
    eval_set=[(Xdv, y_dev)],
    verbose=False,
)

# Dev threshold tuning
probs_dev = xgb.predict_proba(Xdv)[:, 1]
ths = np.linspace(0.2, 0.8, 121)
best_t, best_acc = max(
    ((t, accuracy_score(y_dev, (probs_dev >= t).astype(int))) for t in ths),
    key=lambda x: x[1]
)
print(f"Best trees: {xgb.get_booster().best_iteration + 1}")
print(f"Dev accuracy @t={best_t:.3f}: {best_acc:.4f}")

# Predict test and save
probs_test = xgb.predict_proba(Xte)[:, 1]
pred_pm = np.where(probs_test >= best_t, "+", "-")
out = pd.DataFrame({"id": test["id"], "sentence": test[text_col], "prediction": pred_pm})
out.to_csv(out_path, index=False)
print(f"Wrote {out_path}")
