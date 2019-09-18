import numpy as np

def tst(n_features):
        n_cols = int(np.sqrt(n_features))
        n_rows = int(np.ceil(n_features / n_cols))
        print(f"n_features: {n_features}, n_cols: {n_cols}, n_rows: {n_rows}")

for i in range(8, 37):
        tst(i)

