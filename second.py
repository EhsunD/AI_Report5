import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv('../creditcard.csv')

X_cluster = df.drop(['Class'], axis=1)
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_cluster_scaled)

cluster_analysis = df.groupby('Cluster').mean()

plt.figure(figsize=(8, 6))
sns.scatterplot(x='V1', y='V2', hue='Cluster', data=df, palette='viridis')
plt.title('K-Means Clustering Result')
plt.show()
