import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from Models.utils import drop_pairs_columns

dataset = pd.read_parquet("../../Dataset/Datasets/dataset_pairs_PC.parquet")

df = drop_pairs_columns(dataset)
# print(*dataset.columns.tolist())
df = df.drop(['Rider_gate', 'Opponent_gate'], axis=1)
print(f"Rows before dropna: {len(df)}")
df = df.dropna()
print(f"Rows after dropna {len(df)}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

pc1 = X_pca[:, 0]

correlations = abs(df.corrwith(pd.Series(pc1, index=df.index))).sort_values(ascending=False)
top10 = correlations.head(10)

print("The top 10 most correlated features with PC1:")
print(top10)
plt.figure(figsize=(10, 5))
sns.barplot(y=top10.values, x=top10.index, palette="viridis", legend=False, hue=top10.index)
plt.title("The top 10 most correlated features with PC1")
plt.xlabel("Correlation with z PC1")
plt.xticks(rotation=30)
plt.ylabel("Feature")
plt.ylim(0.7157, 0.7168)
plt.tight_layout()
plt.show()

heat_avg_corrs_rider = correlations[correlations.index.str.match(r"Rider_\d+_heat_avg")]
top10_heat_avg_rider = heat_avg_corrs_rider.head(10)

print("\nThe top 10 most correlated Rider_heat_avg features with PC1:")
print(top10_heat_avg_rider)
plt.figure(figsize=(10, 5))
sns.barplot(y=top10_heat_avg_rider.values, x=top10_heat_avg_rider.index, palette="mako", legend=False, hue=top10_heat_avg_rider.index)
plt.title("The top 10 most correlated Rider_heat_avg features with PC1")
plt.xlabel("Correlation with PC1")
plt.xticks(rotation=30)
plt.ylabel("Rider_heat_avg feature")
plt.ylim(0.7138, 0.7168)
plt.tight_layout()
plt.show()

heat_avg_corrs_opponent = correlations[correlations.index.str.match(r"Opponent_\d+_heat_avg")]
top10_heat_avg_opponent = heat_avg_corrs_opponent.head(10)

print("\nThe top 10 most correlated Opponent_heat_avg features with PC1:")
print(top10_heat_avg_opponent)
plt.figure(figsize=(10, 5))
sns.barplot(y=top10_heat_avg_opponent.values, x=top10_heat_avg_opponent.index, palette="mako", legend=False, hue=top10_heat_avg_opponent.index)
plt.title("The top 10 most correlated Opponent_heat_avg features with PC1")
plt.xlabel("Correlation with PC1")
plt.xticks(rotation=30)
plt.ylabel("Opponent_heat_avg feature")
plt.ylim(0.7138, 0.7168)
plt.tight_layout()
plt.show()

# Take 2

mov_avg_cols = df.columns[df.columns.str.match(r"Rider_\d+_heat_avg")]
df_mov_avg = df[mov_avg_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_mov_avg)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

pc1 = X_pca[:, 0]

correlations = abs(df_mov_avg.corrwith(pd.Series(pc1, index=df_mov_avg.index))).sort_values(ascending=False)

print("\nCorrelations of heat_avg features with PC1:")
print(correlations)
plt.figure(figsize=(12, 6))
hues = range(1, 45)
sns.barplot(y=correlations.values, x=correlations.index , legend=False, hue=hues)
plt.title("Correlations of heat_avg features with PC1")
plt.xlabel("Correlation with PC1")
plt.xticks(rotation=30)
plt.ylabel("Heat_avg Feature")
plt.ylim(0.64, 0.99)
plt.tight_layout()
plt.show()
