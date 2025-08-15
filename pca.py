from utils import load_data, preprocess_features
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = load_data()
X_train, X_test, y_train, y_test = preprocess_features(df, target_column='Type_1')

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

plt.scatter(X_pca[:,0], X_pca[:,1], c=y_train)
plt.title("PCA of Pokémon Stats")
plt.show()
