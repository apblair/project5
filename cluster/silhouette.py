from re import A
import numpy as np
from scipy.spatial.distance import cdist
import collections
from kmeans import *
from utils import *
from sklearn.metrics import silhouette_samples, silhouette_score

class Silhouette:
    def __init__(self, metric: str = "euclidean"):
        """
        inputs:
            metric: str
                the name of the distance metric to use
        """
        self.metric = metric

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features. 

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        c_i = collections.Counter(y)
        d_ij = {label : [sum(cdist(X[index,:].reshape(1,X.shape[-1]), X[y==y[index]], metric=self.metric)[0]) \
            for index in range(X[y==[label]].shape[0])] for label in list(set(y))}
        a_i = {label: [1/(c_i[label]-1) * val for val in v] for label,v in d_ij.items()}
        print(a_i[0])

mat, labels = make_clusters()
# test_mat, test_labels = make_clusters()

k_model = KMeans(k=3)
k_model.fit(mat)
predicted_labels = k_model.predict(mat)

# print(k_model.get_centroids())
# print(k_model.get_error())
# print(k_model.predict(test_mat))

sil = Silhouette()
sil.score(mat, predicted_labels)

# silhouette_values = silhouette_samples(mat, labels)
# print(silhouette_values)
# print(np.allclose(sil.score(mat, predicted_labels),silhouette_values))