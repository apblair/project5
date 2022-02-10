from re import A
import numpy as np
from scipy.spatial.distance import cdist
import collections

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
        
        References
        ----------
        1. https://en.wikipedia.org/wiki/Silhouette_(clustering)
        """
        
        cluster_labels = list(set(y))
        c_i = collections.Counter(y) # number of points belonging to cluster i
        
        # For each cluster, compute the distance between data points i and j in cluster i.
        d_ij = {label : [sum(cdist(X[index,:].reshape(1,X.shape[-1]), X[y==label], metric=self.metric)[0]) \
            for index in list(np.where(y==label)[0])] \
                for label in cluster_labels}

        # For each cluster, compute the mean distance between i and all other data points in the cluster i
        a_i = {label: [1/(c_i[label]-1) * val for val in v] for label,v in d_ij.items()}
        
        # Compute the mean dissimilarity of point i to cluster j, as the mean of the distance from i to all points in cluster j
        b_i = {label: [] for label in cluster_labels}
        for index,label in zip(range(X.shape[0]), y):
            # Find the smallest mean distance of i to all points in any other cluster, of which i is not a member
            j_list = [j for j in cluster_labels if j != label]
            d_ij_list = [(1/c_i[j]) * sum(cdist(X[index,:].reshape(1,X.shape[-1]), X[y==j], metric=self.metric)[0]) for j in j_list] 
            b_i[label].append(min(d_ij_list))
        
        # Compute silhouette values for each data point
        # NOTE: the silhouette dictionary can be used to create silhouette plots (e.g., https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py)
        sil_dict = {label:[] for label in cluster_labels}
        for label in cluster_labels:
            for b,a in zip(b_i[label],a_i[label]):                
                silhouette_numerator =  b-a
                silhouette_denominator = max([b,a])
                sil_dict[label].append(silhouette_numerator/silhouette_denominator)
        
        # Order silhouette values by y
        silhouette_values = []
        for label in y:
            silhouette_values.append(sil_dict[label].pop(0))

        return np.array(silhouette_values)