from utils import load_data, preprocess_features
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = load_data()
X_train, X_test, y_train, y_test = preprocess_features(df, target_column='Type_1')

# Reduce dimensions to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_pca)

plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters)
plt.title("K-Means Clusters")
plt.show()
