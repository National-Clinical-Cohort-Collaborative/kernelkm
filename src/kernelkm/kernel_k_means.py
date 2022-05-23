import numpy as np
import pandas as pd
import sys
import warnings


warnings.filterwarnings('ignore')


class KernelKMeans:
    """
    Perform K-means clustering starting from a similarity (kernel) matrix.
    This type of K-means clustering is not supported out of the box by scikit learn.
    The code was developed for use in the N3C platform.
    """

    def __init__(self, datamat: np.ndarray, patient_id_list: list, max_iter=100):
        """
        datamat: a symmetric matrix (as a numpy array) with patient-patient similarities
        patient_id_list: a list of patient ids corresponding to the rows (and columns) of datamat
        max_iter: maximum number of iteration to do when refining clusters [100]
        """
        if not isinstance(datamat, np.ndarray):
            raise ValueError("datamat parameter needs to be an np.ndarray")
        if datamat.ndim != 2:
            raise ValueError("datamat parameter needs to be 2 dimensional array")
        if not isinstance(patient_id_list, list):
            raise ValueError("patient_id_list needs to be a python list")
        shape = datamat.shape
        if shape[0] != shape[1]:
            raise ValueError("datamat needs to be a square matrix")
        if shape[0] != len(patient_id_list):
            raise ValueError("datamat needs to have same dimension as patient_id_list")
        self._matrix = datamat
        self._pat_id_list = patient_id_list
        self._max_iter = max_iter

    def calculate(self, k):
        """Run k means clustering for the given k (number of clusters)

        """

        if not isinstance(k, int):
            raise ValueError("Must call this function with one argument - an integer k for the number of clusters")
        centroids = self.plus_plus(k) # _init_centroids(k)
        errors = []
        diff = True
        i = 0
        centroid_assignments = []

        while diff:
            i += 1
            centroid_assignments, centroid_errors = self._assign_to_centroid(centroids)
            errors.append(centroid_errors)
            centroids_new = self._adjust_centroids(centroid_assignments)
            if centroids.equals(centroids_new):
                diff = False
            else:
                centroids = centroids_new
            if i == self._max_iter:
                print("Reaching maximum allowed iterations ({self._max_iter}), terminating optimization loop")
                break
        return centroids, centroid_assignments, errors

    def plus_plus(self, k):
        """Create cluster centroids using the k-means++ algorithm.

        This method makes an effort to choose centroids that are far apart from one another when initializing
        the clustering. This in theory reduces the chance of very bad clustering that results from poorly chosen
        initial centroids. Inspiration from here:
        https://stackoverflow.com/questions/5466323/how-could-one-implement-the-k-means-algorithm

        Parameters
        ----------
        ds : numpy array
            The dataset to be used for centroid initialization.
        k : int
            The desired number of clusters for which centroids are required.
        Returns
        -------
        centroids : numpy array
            Collection of k centroids as a numpy array.
        """

        ds = self._matrix
        centroids = [ds[0]]

        for _ in range(1, k):
            dist_sq = np.array([min([np.inner(c-x, c-x) for c in centroids]) for x in ds])
            probs = dist_sq/dist_sq.sum()
            cumulative_probs = probs.cumsum()
            r = np.random.rand()

            for j, p in enumerate(cumulative_probs):
                if r < p:
                    i = j
                    break
            centroids.append(ds[i])
        return pd.DataFrame(centroids, columns=self._pat_id_list)

    def _init_centroids(self, k):
        """
        initialize centroids to uniform random values between the minimum (0.0) and maximum of all similarity values
        """
        centroid_min = 0.0
        centroid_max = self.get_max_value()
        n = self.get_patient_count()
        centroids = []  # a list of np.ndarray's
        for centroid in range(k):
            centroid = np.random.uniform(centroid_min, centroid_max, n)
            centroids.append(centroid)
            if n != len(centroid):
                raise ValueError("Problem constructing centroid")
        centroids = pd.DataFrame(centroids, columns=self._pat_id_list)
        return centroids

    def _assign_to_centroid(self, centroids):
        n_patients = self.get_patient_count()
        centroids_assigned = []
        centroid_errors = []
        k = len(centroids)

        for pat in range(n_patients):
            min_centroid_error = sys.float_info.max
            closest_centroid_idx = -1
            for centroid in range(k):
                # centroids.iloc retrieves a pandas series
                # and seld._matrix[pat, :] retrieves a ndarray
                # both represent vectors of similarities
                patient_a = centroids.iloc[centroid, :].to_numpy()
                patient_b = self._matrix[pat, :]
                if len(patient_a) != len(patient_b):
                    raise ValueError(f"Unqual lengths - centroid {centroid}: {len(patient_a)} and patient {pat}: {len(patient_b)}")
                error = np.sqrt(np.sum((patient_a - patient_b)**2))
                if error < min_centroid_error:
                    min_centroid_error = error
                    closest_centroid_idx = centroid
            if closest_centroid_idx < 0:
                #  if this happens, there is probably an error that the user needs to know about
                #  and so we should stop execution.
                raise ValueError(f"Failed to assign patient {pat} to centroid (should never happen)")
            centroids_assigned.append(closest_centroid_idx)
            centroid_errors.append(min_centroid_error)

        return centroids_assigned, centroid_errors

    def calculate_sse(self, centroids, centroid_assignments):
        """Calculates the SSE for a given clustering

        The SSE is the sum of the square of the Euclidean distance from each data points (here a
        patient) to the centroid of the cluster to which the data point has been assigned.
        """
        n_patients = self.get_patient_count()
        sse = 0
        for pat_idx in range(n_patients):
            c = centroid_assignments[pat_idx]
            assigned_centroid = centroids.iloc[c, :].to_numpy()
            patient_vector = self._matrix[pat_idx, :]
            error = np.sum((assigned_centroid - patient_vector)**2)
            sse += error
        return sse

    def _adjust_centroids(self, centroid_assignments):
        """
        centroid_assignments - a list of integers with the same length as the number of patients
                               each entry represents the centroid to which the patient has been assigned
        return - a DataFrame with the new centroids that correspond to the patient assignments.
        """
        if not isinstance(centroid_assignments, list):
            raise ValueError("centroids argument must be a list")
        mat = self._matrix.copy()
        df = pd.DataFrame(mat)
        df['cluster'] = centroid_assignments

        new_centroids = pd.DataFrame(df).groupby(by='cluster').mean()
        return new_centroids

    def get_patient_count(self) -> int:
        """Helper function to return the number of patients in the patient list of the KernelKMeans instance
        """
        return len(self._pat_id_list)

    def get_max_value(self) -> float:
        """Return the largest similarity observed in the entire patient-patient similarity matrix

        return:  maximum over all data points
        """
        return self._matrix.max().max()
