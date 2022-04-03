import numpy as np
from collections import defaultdict
import pandas as pd
from .kernel_k_means import KernelKMeans

class Elbow:
    """
    Elbow heuristic for getting a good k kernel k-means algorithm
    """

    def __init__(self, datamat, patient_id_list, max_k=10, max_iter=100):
        """
        datamat: matrix of pairwise similarities
        patient_id_list: list of patient ids that corresponds to datamat
        max_k: try from 1 up to this k number of clusters
        """
        if not isinstance(datamat, np.ndarray):
            raise ValueError("Must pass datamat as np.ndarray")
        shape = datamat.shape
        if shape[0] != shape[1]:
            raise ValueError("datamat needs to be a square matrix")
        if shape[0] != len(patient_id_list):
            raise ValueError("datamat needs to have same dimension as patient_id_list")
        self._matrix = datamat
        self._pat_id_list = patient_id_list
        self._max_k = max_k
        self._max_iter = max_iter

    def get_sse_for_up_to_k_clusters(self) -> pd.DataFrame:
        sse = defaultdict(int)
        results = []
        kkm = KernelKMeans(self._matrix, self._pat_id_list, self._max_iter)
        # to test up to self._max_k we need the following range!
        for k in range(1, self._max_k + 1):
            centroids, centroid_assignments, errors = kkm.calculate(k=k)
            sse = kkm.calculate_sse(centroids, centroid_assignments)
            d = {"k": k, "sse": sse}
            results.append(d)
        return pd.DataFrame(results)