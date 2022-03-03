from unittest import TestCase
import numpy as np
from kernelkm import KernelKMeans, GapStat
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.spatial import distance_matrix
import pandas as pd


class TestKMeans(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # Make 6x6 matrix
        cls._mat = np.array([[10, 5, 7, 1, 1, 1],  # Patient 1 similarities
                             [5, 10, 4, 1, 1, 1],  # Patient 2 similarities
                             [7, 4, 10, 1, 1, 1],  # Patient 3 similarities
                             [1, 1, 1, 10, 5, 5],  # Patient 4 similarities
                             [1, 1, 1, 5, 10, 5],  # Patient 5 similarities
                             [1, 1, 1, 5, 5, 10]])  # Patient 6 similarities
        cls._labels = ["p1", "p2", "p3", "p4", "p5", "p6"]
        cls._kkm = KernelKMeans(datamat=cls._mat, patient_id_list=cls._labels)
        #np.random.seed(24)

    def test_ctor(self):
        kkm = self._kkm
        self.assertIsNotNone(kkm)

    def test_get_maximum_value(self):
        kkm = self._kkm
        self.assertEqual(10, kkm.get_max_value())

    def test_get_patient_count(self):
        kkm = self._kkm
        self.assertEqual(6, kkm.get_patient_count())

    def test_clustering(self):
        kkm = self._kkm
        centroids, centroid_assignments, error = kkm.calculate(k=2)
        print(centroid_assignments)
        print(centroids)
        self.assertEqual(centroid_assignments[0], centroid_assignments[1])

    def test_gap_stat(self):
        gstat = GapStat(datamat=self._mat, patient_id_list=self._labels)
        k, _, _ = gstat.calculate_good_k()
        self.assertEqual(2, k)

    def test_on_blob_data(self):
        correct_num_clusters = 3
        patient_IDs = ["patient" + str(i) for i in range(3)]
        X, correct_clusters = make_blobs(n_samples=100, n_features=2,
                                         centers=correct_num_clusters, cluster_std=.8,)[0]

        # turn X in sim matrix to pass to kernelkm 
        # Original code from OP, slightly reformatted
        # DF_var = pd.DataFrame.from_dict({
        #     "s1":[1.2,3.4,10.2],
        #     "s2":[1.4,3.1,10.7],
        #     "s3":[2.1,3.7,11.3],
        #     "s4":[1.5,3.2,10.9]
        # }).T
        # DF_var.columns = ["g1","g2","g3"]

        # Whole similarity algorithm in one line
        X_sim = pd.DataFrame(
            1 / (1 + distance_matrix(X, X)),
            columns=patient_IDs, index=patient_IDs
        )

        #           g1        g2        g3
        # g1  1.000000  0.215963  0.051408
        # g2  0.215963  1.000000  0.063021
        # g3  0.051408  0.063021  1.000000

        plt.figure(figsize=(12, 3))
        for k in range(1, 6):
            # kmeans = KMeans(n_clusters=k)
            # a = kmeans.fit_predict(X)
            kmeans = KernelKMeans(datamat=np.array(X_sim), patient_id_list=patient_IDs)
            centroids, centroid_assignments, errors = kmeans.calculate(k)
            plt.subplot(1, 5, k)
            plt.scatter(X[:, 0], X[:, 1], c=centroid_assignments)
            plt.xlabel('k='+str(k))
        plt.tight_layout()
        plt.show()




