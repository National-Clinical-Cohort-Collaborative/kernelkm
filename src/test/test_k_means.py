import numpy as np
from unittest import TestCase
from kernelkm import KernelKMeans


class TestKMeans(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # Make 6x6 matrix
        cls._mat = np.array([[0, 5, 7, 1, 1, 1],  # Patient 1 similarities
                             [5, 0, 4, 1, 1, 1],  # Patient 2 similarities
                             [7, 4, 0, 1, 1, 1],  # Patient 3 similarities
                             [1, 1, 1, 0, 5, 5],  # Patient 4 similarities
                             [1, 1, 1, 5, 0, 5],  # Patient 5 similarities
                             [1, 1, 1, 5, 5, 0]])  # Patient 6 similarities
        cls._labels = ["p1", "p2", "p3", "p4", "p5", "p6"]
        cls._kkm = KernelKMeans(datamat=cls._mat, patient_id_list=cls._labels)
        np.random.seed(24)

    def test_ctor(self):
        kkm = self._kkm
        self.assertIsNotNone(kkm)

    def test_get_maximum_value(self):
        kkm = self._kkm
        self.assertEqual(7, kkm.get_max_value())

    def test_get_patient_count(self):
        kkm = self._kkm
        self.assertEqual(6, kkm.get_patient_count())

    def test_clustering(self):
        kkm = self._kkm
        centroids, centroid_assignments, error = kkm.calculate(k=2)
        print(centroids)
        self.assertEqual(centroid_assignments[0], centroid_assignments[1])






