# write your silhouette score unit tests here
import numpy as np
from scipy.spatial.distance import cdist
import cluster
import pytest

from sklearn.metrics import silhouette_samples, silhouette_score


def test_silhouette():
    """
    Unit test of silhouette scores for each of the observations.
    
    NOTE: Using sklearn.metrics.silhouette_samples as ground truth. (NOT sklearn.metrics.silhouette_score)

    References
    ----------
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_samples.html 
    """
    # Generate cluster data
    mat, labels = cluster.utils.make_clusters()

    # Fit model and predict labels
    k_model = cluster.KMeans(k=3)
    k_model.fit(mat)
    predicted_labels = k_model.predict(mat)

    # Calculate silhouette scores and check against sklearn
    bmi203_proj5_sihouette_values = cluster.Silhouette().score(mat, predicted_labels)
    sklearn_silhouette_values = silhouette_samples(mat, predicted_labels)
    assert np.allclose(bmi203_proj5_sihouette_values, sklearn_silhouette_values)