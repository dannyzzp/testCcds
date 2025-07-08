import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
def runPCA():
    df = pd.read_csv("~/projects/testCcds/data/raw/creditcard.csv")
    X = df.drop(columns=['Class'])
    pca = PCA(n_components='mle', svd_solver = 'full')
    pca.fit(X)
    X_pca = pd.DataFrame(pca.transform(X))
    df_pca = pd.concat([X_pca, df['Class']], axis=1)
    return df_pca
