import pandas as pd
from sklearn.preprocessing import OneHotEncoder

train = pd.read_csv("hw1-data/income.train.5k.csv", header=0, skipinitialspace=True, na_filter=False)

feature_cols = ["age","sector","edu","marriage","occupation","race","sex","hours","country"]
Xtr = train[feature_cols].astype(str)

enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
enc.fit(Xtr)
Xtr_bin = enc.fit_transform(Xtr)

names = enc.get_feature_names_out(feature_cols)
print("Sklearn Transformed:", Xtr_bin.shape[1])               # -> 230
print("Training set:", train[feature_cols].nunique().sum())  # -> 230