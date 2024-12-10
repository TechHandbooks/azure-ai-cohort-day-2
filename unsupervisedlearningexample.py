from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
data = iris.data

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data)

# Cluster labels
print("Cluster labels:", kmeans.labels_)

# Visualize clusters (using first two features)
plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title("KMeans Clustering of Iris Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
