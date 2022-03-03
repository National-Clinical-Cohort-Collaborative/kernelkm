import numpy as np
import copy
from .kernel_k_means import KernelKMeans


class GapStat:
    """
    Gap statistic for kernel k-means algorithm
    """

    def __init__(self, datamat, patient_id_list, max_k=10, B=4, max_iter=100):
        """
        B: number of permutations/randomizations for gapstat
        max_iter: for K means
        """
        shape = datamat.shape
        if shape[0] != shape[1]:
            raise ValueError("datamat needs to be a square matrix")
        if shape[0] != len(patient_id_list):
            raise ValueError("datamat needs to have same dimension as patient_id_list")
        self._matrix = datamat
        self._pat_id_list = patient_id_list
        self._max_iter = max_iter
        self._max_k = max_k
        self._B = B

    def calculate_good_k(self):

        """
        data check in KernelKMeans class!
        """
        kkm = KernelKMeans(self._matrix, self._pat_id_list, self._max_iter)
        gap_stat = [0]
        s_stat = [0]
        # to test up to self._max_k we need the following range!
        for k in range(1, self._max_k + 1):
            centroids, centroid_assignments, errors = kkm.calculate(k=k)
            W_k_observed = self._calculate_W_k(self._matrix, centroids, centroid_assignments)
            W_k_expectation, s_k = self._get_avg_permuted_W_k(k)
            gap_k = W_k_expectation - W_k_observed
            gap_stat.append(gap_k)
            s_stat.append(s_k)
            # check if gap(k-1) \geq gap(k) - s_{k}
            if k > 1:
                if gap_stat[k-1] - gap_stat[k] + s_stat[k] > 0:
                    return k-1, s_stat, gap_stat
        return self._max_k, s_stat, gap_stat  # if we get here we do not have great clusters

    def _get_avg_permuted_W_k(self, this_k: int):
        w_k_estimate = []
        for i in range(self._B):
            randomized_M = self._get_permuted_matrix()
            kkm = KernelKMeans(randomized_M, self._pat_id_list, self._max_iter)
            centroids, centroid_assignments, errors = kkm.calculate(k=this_k)
            w_k_star = self._calculate_W_k(randomized_M, centroids, centroid_assignments)
            w_k_estimate.append(w_k_star)
        s_dk = np.std(w_k_estimate)
        s_k = s_dk * np.sqrt(1+1/self._B)
        return np.mean(w_k_estimate), s_k

    def _get_permuted_matrix(self):
        """
        permute entries of matrix, reshape 2d to 1d and use np.permutation and reshape back
        """
        N = len(self._pat_id_list)
        mat = copy.deepcopy(self._matrix)
        mat = mat.reshape(N * N)
        np.random.shuffle(mat)
        mat = mat.reshape((N,N))
        return mat

    def _calculate_D_r(self, matrix, centroid, assigned_to_centroid):
        """
        sum of pairwise distances
        """
        D_j = 0
        n_r = np.count_nonzero(assigned_to_centroid)
        for i, i_bool in enumerate(assigned_to_centroid):
            for j, j_bool in enumerate(assigned_to_centroid):
                if i_bool and j_bool:
                    d_ij = np.sqrt(np.sum((matrix[i,]-matrix[j,])**2))
                    D_j += d_ij
        return D_j/(2*n_r)

    def _calculate_W_k(self, matrix, centroids, centroid_assignments):
        k = len(centroids)
        W_k = 0
        for i in range(k):
            centroid = centroids[i]
            # check which patients are assigned to centroid i and make a boolean matrix
            # to indicate whether a patient belongs to centroid i (True) or not (False)
            assigned = [x == i for x in centroid_assignments]
            D_r = self._calculate_D_r(matrix, centroid, assigned)
            W_k += D_r
        return W_k
