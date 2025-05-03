import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pyod.models.pca import PCA
from sklearn.manifold import TSNE

# === Load CSVs ===
dql_log = pd.read_csv("DQLogger.csv")
mwd_log = pd.read_csv("MWDLogger.csv")

# === Define numeric features (for PCA only) ===
dql_columns = [
    'drilledMeters', 'penRate', 'overDrill',
    'spatialInaccuracy', 'bitDiameter'
]

mwd_columns = [
    'AvePenetrRate', 'AvePercPressure', 'AveFeedPressure',
    'AveDampPressure', 'AveRotPressure', 'AveFlushPressure'
]

# === Clean and split DQL data ===
dql_numeric = dql_log[dql_columns].dropna().copy()
dql_metadata = dql_log[['holeName', 'startHoleTime']] 

# === Clean and split MWD data ===
mwd_numeric = mwd_log[mwd_columns].dropna().copy()
mwd_metadata = mwd_log[['Hole ID', 'Start Hole Time']].loc[mwd_numeric.index].reset_index(drop=True)

# === PCA on DQL ===
pca_dql = PCA()
pca_dql.fit(dql_numeric.values)
dql_numeric['anomaly_score'] = pca_dql.decision_scores_
dql_numeric['anomaly_label'] = pca_dql.labels_

# === PCA on MWD ===
pca_mwd = PCA()
pca_mwd.fit(mwd_numeric.values)
mwd_numeric['anomaly_score'] = pca_mwd.decision_scores_
mwd_numeric['anomaly_label'] = pca_mwd.labels_

# === Combine MWD with metadata ===
mwd_full = pd.concat([mwd_metadata, mwd_numeric.reset_index(drop=True)], axis=1)

# === Save MWD anomaly results ===
mwd_full.to_csv("MWD_Anomaly_Results.csv", index=False)
mwd_anomalies_only = mwd_full[mwd_full['anomaly_label'] == 1]
mwd_anomalies_only.to_csv("MWD_Only_Anomalies.csv", index=False)

# === Save DQL anomaly results === #
dql_full = pd.concat([dql_metadata.reset_index(drop=True), dql_numeric.reset_index(drop=True)], axis=1)
dql_anomalies_only = dql_full[dql_full['anomaly_label'] == 1]
dql_anomalies_only.to_csv("DQL_Only_Anomalies.csv", index=False)


# === t-SNE Visualization for DQL ===
tsne_dql = TSNE(n_components=2, random_state=42)
dql_2d = tsne_dql.fit_transform(dql_numeric[dql_columns])
dql_vis = pd.DataFrame(dql_2d, columns=['TSNE1', 'TSNE2'])
dql_vis['anomaly_label'] = dql_numeric['anomaly_label']

plt.figure(figsize=(10, 6))
sns.scatterplot(data=dql_vis, x='TSNE1', y='TSNE2', hue='anomaly_label', palette={0: 'blue', 1: 'red'})
plt.title("t-SNE Projection of DQL Anomalies (5D → 2D)")
plt.grid(True)
plt.tight_layout()
plt.show()

# === t-SNE Visualization for MWD ===
tsne_mwd = TSNE(n_components=2, random_state=42)
mwd_2d = tsne_mwd.fit_transform(mwd_numeric[mwd_columns])
mwd_vis = pd.DataFrame(mwd_2d, columns=['TSNE1', 'TSNE2'])
mwd_vis['anomaly_label'] = mwd_numeric['anomaly_label']

plt.figure(figsize=(10, 6))
sns.scatterplot(data=mwd_vis, x='TSNE1', y='TSNE2', hue='anomaly_label', palette={0: 'blue', 1: 'red'})
plt.title("t-SNE Projection of MWD Anomalies (6D → 2D)")
plt.grid(True)
plt.tight_layout()
plt.show()

