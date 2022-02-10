import numpy as np
from scipy.spatial.distance import cdist
import cluster
import pytest

from sklearn.cluster import KMeans

def test_kmeans():
    """
    Unit test for kmeans implementation.
    """
    # Generate cluster data
    mat, labels = cluster.utils.make_clusters()

    # Fit model and predict labels
    bmi_proj5_k_model = cluster.KMeans(k=3)
    bmi_proj5_k_model.fit(mat)
    bmi_proj5_k_model_centroids = set(np.round(np.concatenate(bmi_proj5_k_model.get_centroids()),decimals=3))

    # Set sklearn KMean model paramters to cluster.KMeans default parameters
    sklearn_k_model = KMeans(n_clusters=3, init='random', n_init=bmi_proj5_k_model.max_iter, tol=bmi_proj5_k_model.tol, random_state=42)
    sklearn_k_model.fit(mat)
    sklearn_k_centroids = set(np.round(np.concatenate(sklearn_k_model.cluster_centers_),decimals=3))

    assert len(bmi_proj5_k_model_centroids.union(sklearn_k_centroids)) == 6