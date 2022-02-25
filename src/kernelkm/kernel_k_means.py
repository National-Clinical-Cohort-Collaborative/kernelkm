import numpy as np 
import pandas as pd 
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
        centroids_assigned = self._init_centroids(k)
        error = []
        compr = True
        i = 0

        while compr:
            centroids_assigned, centroid_errors = self._assign_to_centroid(centroids_assigned)
            centroids_assigned = self._adjust_centroids(centroids_assigned)




    def _init_centroids(self, k):
        """
        initialize centroids to uniform random values between the minimum (0.0) and maximum of all similarity values
        """
        centroid_min = 0.0 
        centroid_max = self._matrix.max().max() # maximum over all data points
        n = self._matrix.shape[0] # number of patients
        centroids = []
        for centroid in range(k):
            centroid = np.random.uniform(centroid_min, centroid_max, n)
            centroids.append(centroid)
        centroids = pd.DataFrame(centroids, columns=self._pat_id_list)
        return centroids

    def _sum_of_squared_error(self, vec_a, vec_b):
        """
        vec_a and vec_b must both be np arrays of floats
        this is the SSE
        """
        return np.square(np.sum(a-b)**2)

    def _assign_to_centroid(self, centroids):
        n_patients = len(self._pat_id_list)
        centroids_assigned = []
        centroid_errors = []
        k = centroids.shape[0]

        for pat in range(n_patients):
            errors = np.array([])
            for centroid in range(k):
                vec_a
                error = self._sum_of_squared_error(centroids.iloc[centroid, :2], self._matrix[pat, :2])
                errors = np.append(errors, error)
            
            clostest_centroid = np.where(errors == np.amin(error))[0].tolist()[0]
            centroid_err = np.amin(errors)

            centroids_assigned.append(clostest_centroid)
            centroid_errors.append(centroid_err)

        return centroids_assigned, centroid_errors


    


