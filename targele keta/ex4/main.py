from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import seaborn as sns
from sklearn.cluster import KMeans
X, y = make_blobs(n_samples=100, centers=5, random_state=101)
elbow = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 101)
    kmeans.fit(X)
    elbow.append(kmeans.inertia_)
sns.lineplot(range(1, 20), elbow, color='blue')
plt.rcParams.update({'figure.figsize':(10,7.5), 'figure.dpi':100})
plt.title('ELBOW METHOD')
plt.show()
