import numpy as np
from scipy.spatial.distance import cdist
from utils import *

class KMeans:
    def __init__(
            self,
            k: int,
            metric: str = "euclidean",
            tol: float = 1e-6,
            max_iter: int = 100):
        """
        inputs:
            k: int
                the number of centroids to use in cluster fitting
            metric: str
                the name of the distance metric to use
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        assert k > 0, "Error: The number of centroids must be greater than 0."
        self.k = k
        self.metric = metric
        self.tol = tol
        self.max_iter = max_iter
        
    def _init_centroids(self):
        """
        Compute the initial centroids

        References:
        -----------
        1. https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        2. https://www.askpython.com/python/examples/k-means-clustering-from-scratch
        """
        self._centroids = self._mat[np.random.choice(self._mat.shape[0], self.k, replace=False),:]
        self._distances = cdist(self._mat, self._centroids, self.metric)
        self._labels = np.argmin(self._distances, axis=1)
    
    def _calculate_mse(self):
        """
        """
        points_to_centroid = np.array(list(map(lambda label: self._centroids[label], self._labels)))
        distances = np.diag(cdist(self._mat, points_to_centroid, metric=self.metric))
        distances_squared = np.square(distances)
        mse = distances_squared.mean()
        return mse

    def fit(self, mat: np.ndarray):
        """
        fits the kmeans algorithm onto a provided 2D matrix

        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        
        References
        ----------
        1. https://blog.paperspace.com/speed-up-kmeans-numpy-vectorization-broadcasting-profiling/
        """
        assert self.k < mat.shape[0], "Error: The number of centroids must be less than the number of observations."
        if mat.ndim == 1:
            print("Warning: Reshaping 1D numpy array (-1,1).")
            print("Warning: Consider an alternative algorithm like KDE for one dimensional data.")
            mat = mat.reshape(-1,1) 

        self._mat = mat
        self._mse = []
        for iter_step in range(self.max_iter):
            self._init_centroids()
            for k_cluster in range(self.k):
                self._centroids[k_cluster,:] = np.mean(self._mat[self._labels == k_cluster,:],axis = 0)
            self._mse.append(self._calculate_mse())

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        predicts the cluster labels for a provided 2D matrix

        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """

    def get_error(self) -> float:
        """
        returns the final squared-mean error of the fit model

        outputs:
            float
                the squared-mean error of the fit model
        """

    def get_centroids(self) -> np.ndarray:
        """
        returns the centroid locations of the fit model

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """

mat, labels = make_clusters()
# print(mat)
# print(labels)

x = np.random.random(100)
KMeans(k=3).fit(x)