import numpy as np
from scipy.spatial.distance import cdist

class KMeans:
    def __init__(
            self,
            k: int,
            metric: str = "euclidean",
            tol: float = 1e-6,
            max_iter: int = 100,
            seed: int = 42):
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
        np.random.seed(seed)
        assert k > 0, "Error: The number of centroids must be greater than 0."
        assert k > 1, "Error: Setting k=1 means every point belongs to the same cluster."
        self.k = k
        self.metric = metric
        self.tol = tol
        self.max_iter = max_iter
    
    def _check_input_mat(self, mat: np.ndarray) -> np.ndarray: 
        """
        Check if the number of centroids are less than the number of observations in the input matrix.
        Check if the input matrix is one dimensional

        inputs: 
            mat: np.ndarray
        outputs:
            mat: np.ndarray
        """
        assert self.k < mat.shape[0], "Error: The number of centroids must be less than the number of observations."

        if mat.ndim == 1:
            print("Warning: Reshaping 1D numpy array (-1,1).")
            print("Warning: Consider an alternative algorithm like KDE for one dimensional data.")
            mat = mat.reshape(-1,1) 

        return mat

    def _find_nearest_centroids(self,  mat: np.ndarray) -> np.ndarray:
        """
        Find the nearest centroids for each point.
        
        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        output:
            np.ndarray
                A 1D array specifying the class labels

        References:
        -----------
        1. https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        2. https://www.askpython.com/python/examples/k-means-clustering-from-scratch
        """
        self._distances = cdist(mat, self._centroids, self.metric)
        return np.argmin(self._distances, axis=1)
    
    def _calculate_mse(self, mat: np.ndarray) -> float:
        """
        Calculate the mean squared error (mse) of the centroid distances.
        
        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        points_to_centroid = np.array(list(map(lambda label: self._centroids[label], self._labels)))
        distances = np.diag(cdist(mat, points_to_centroid, metric=self.metric))
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
        mat = self._check_input_mat(mat)
        
        # Initialize random centroids
        self._centroids = np.random.rand(self.k, mat.shape[1])
        self._labels = self._find_nearest_centroids(mat)
        
        # Fit model
        self._mse = []
        for iter_step in range(self.max_iter):
            for k_cluster in range(self.k):
                # Find the mean of the cluster for the new centroid
                self._centroids[k_cluster,:] = np.mean(mat[self._labels == k_cluster,:],axis = 0)
        
            # Find the nearest centroid for each point and compute the MSE
            self._labels = self._find_nearest_centroids(mat)
            self._mse.append(self._calculate_mse(mat))

            # Check clustering stability against previous MSE during optimization to end the model fit
            if iter_step > 0 and abs(self._mse[iter_step] - self._mse[iter_step-1]) <= self.tol:
                self._mse = self._mse[iter_step]
                break

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
        test_mat = self._check_input_mat(mat)
        predicted_labels = self._find_nearest_centroids(test_mat)
        return predicted_labels

    def get_error(self) -> float:
        """
        returns the final squared-mean error of the fit model

        outputs:
            float
                the squared-mean error of the fit model
        """
        return self._mse

    def get_centroids(self) -> np.ndarray:
        """
        returns the centroid locations of the fit model

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        return self._centroids
