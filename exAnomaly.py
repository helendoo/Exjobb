# anomaly_module.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pyod.models.pca import PCA
from sklearn.manifold import TSNE

def load_and_clean_data(dql_path: str, mwd_path: str):
    dql_log = pd.read_csv(dql_path)
    mwd_log = pd.read_csv(mwd_path)

    dql_columns = ['drilledMeters', 'penRate', 'overDrill', 'spatialInaccuracy', 'bitDiameter']
    mwd_columns = ['avePenetrRate', 'avePercPressure', 'aveFeedPressure', 'aveDampPressure', 'aveRotPressure', 'aveFlushPressure']

    dql_numeric = dql_log[dql_columns].dropna().copy()
    mwd_numeric = mwd_log[mwd_columns].dropna().copy()
    mwd_metadata = mwd_log[['holeID', 'startHoleTime']].loc[mwd_numeric.index].reset_index(drop=True)

    return dql_numeric, mwd_numeric, mwd_metadata, dql_columns, mwd_columns

def run_pca(data: pd.DataFrame):
    # Drop columns with zero variance (constant features)
    data = data.loc[:, data.std() > 0]

    # Drop rows with any remaining NaNs or Infs
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    if data.empty:
        raise ValueError("No valid data left after cleaning for PCA anomaly detection.")

    pca = PCA()
    pca.fit(data.values)
    data['anomaly_score'] = pca.decision_scores_
    data['anomaly_label'] = pca.labels_
    return data


def save_results(data: pd.DataFrame, metadata: pd.DataFrame, path: str):
    full_data = pd.concat([metadata, data.reset_index(drop=True)], axis=1)
    full_data.to_csv(path, index=False)
    return full_data
