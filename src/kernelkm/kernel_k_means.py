import numpy as np
import pandas as pd
import sys
import warnings


warnings.filterwarnings('ignore')


class KernelKMeans:
    """
    Perform K-means clustering starting from a similarity (kernel) matrix.
    This type of K-means clustering is not supported out of the box by scikit learn
    The code was developed for the N3C platform
    """

    def __init__(self, datamat, patient_id_list):
        """
        datamat is a symmetric matrix with patient-patient similarities
        patient_id_list is a list of patient ids corresponding to the rows (and columns) of datamat
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

    def calculate(self, k):
        if not isinstance(k, int):
            raise ValueError("Must call this function with one argument - an integer k for the number of clusters")
        centroids = self._init_centroids(k)
        errors = []
        diff = True
        i = 0

        while diff:
            print(f"Round {i}")
            i += 1
            centroid_assignments, centroid_errors = self._assign_to_centroid(centroids)
            errors.append(centroid_errors)
            centroids_new = self._adjust_centroids(centroid_assignments)
            if np.count_nonzero(centroids-centroids_new) == 0:
                diff = 0
            else:
                centroids = centroids_new
        return centroids, errors

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
        centroids = pd.DataFrame(centroids, columns=self._pat_id_list)
        return centroids

    def _sum_of_squared_error(self, vec_a, vec_b):
        """
        vec_a and vec_b must both be np arrays of floats
        """
        return np.square(np.sum(vec_a-vec_b)**2)

    def _assign_to_centroid(self, centroids):
        n_patients = self.get_patient_count()
        centroids_assigned = []
        centroid_errors = []
        k = len(centroids)

        for pat in range(n_patients):
            min_centroid_error = sys.float_info.max
            closest_centroid_idx = -1
            for centroid in range(k):
                # centroids.iloc and seld._matrix[pat, :] both retrieve vectors of similarities
                error = self._sum_of_squared_error(centroids.iloc[centroid, :], self._matrix[pat, :])
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

    def _adjust_centroids(self, centroid_assignments):
        """
        centroid_assignments - a list of integers with the same length as the number of patients
                               each entry represents the centroid to which the patient has been assigned
        return - a DataFrame with the new centroids that correspond to the patient assignments.
        """
        if not isinstance(centroid_assignments, list):
            raise ValueError("centroids argument must be a list")

        new_centroids = pd.DataFrame(self._matrix).groupby(by=centroid_assignments).mean()
        return new_centroids

    def get_patient_count(self):
        return len(self._pat_id_list)

    def get_max_value(self):
        """
        return:  maximum over all data points
        """
        return self._matrix.max().max()
