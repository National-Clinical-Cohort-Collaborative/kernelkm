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
        k, _ = gstat.calculate_good_k()
        self.assertEqual(2, k)

    def test_on_blob_data(self):
        for this_k in range(2, 7):  # check several correct cluster numbers
            num_patients = 100
            patient_IDs = ["patient" + str(i) for i in range(num_patients)]
            X, correct_cluster_assignments = make_blobs(n_samples=num_patients, n_features=4,
                                                        centers=this_k, cluster_std=.05,)
            # Whole similarity algorithm in one line
            X_sim = pd.DataFrame(1 / (1 + distance_matrix(X, X)), columns=patient_IDs, index=patient_IDs)

            gstat = GapStat(datamat=X_sim.to_numpy(), patient_id_list=patient_IDs)
            inferred_k, _ = gstat.calculate_good_k()
            print(f"this_k: {this_k} inferred_k: {inferred_k}")
            self.assertEqual(this_k, inferred_k)

    def test_gap_stat_one_cluster(self):
        gstat = GapStat(datamat=self._mat, patient_id_list=self._labels, max_k=1)
        k, g_stat = gstat.calculate_good_k()
        self.assertEqual(1, k)
        # The following cannot be expected given the plus plus algorithm,
        # to initiate centroids and the small amount of data for adjusting 
        # centroids in the K Means algorithm
        # self.assertTrue(g_stat[0] == 0)

    def test_gap_stat_ten_clusters(self):
        gstat = GapStat(datamat=self._mat, patient_id_list=self._labels, max_k=10)
        k, g_stat = gstat.calculate_good_k(do_all_k=True)
        self.assertEqual(10, len(g_stat))
