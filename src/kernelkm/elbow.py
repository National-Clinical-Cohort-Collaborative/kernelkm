import numpy as np
from collections import defaultdict
import pandas as pd
from .kernel_k_means import KernelKMeans


class Elbow:
    """
    Elbow heuristic for choosing a good k (number of clusters) when applying kernel k-means algorithm

    K means clustering requires a k parameter that specifies how many clusters should be used during
    clustering. In many cases, we would like a data-driven way of choosing the appropriate k.

    The elbow method does just this. To use the elbow method, we choose a range of k's to investigate
    (say, 1 through 10), and for each k, we cluster the data, then determine the sum squared errors (SSE)
    for each k. The SSE is the sum of the square of the Euclidean distance from each data points (here a
    patient) to the centroid of the cluster to which the data point has been assigned. We then choose 
    the k at which the SSE begins to level off.
    """

    def __init__(self, datamat, patient_id_list, max_k=10, max_iter=100):
        """
        datamat: matrix of pairwise similarities
        patient_id_list: list of patient ids that corresponds to datamat
        max_k: try from 1 up to this k number of clusters
        max_iter: maximum number of iterations to refine clusters
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
        """Get SSE for a range of k values, used to generate the data to apply the Elbow method

        """
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