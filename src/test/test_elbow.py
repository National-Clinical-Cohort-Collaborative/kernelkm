from unittest import TestCase
import numpy as np
from kernelkm import KernelKMeans, Elbow
import pandas as pd


class TestElbow(TestCase):

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

    def test_get_4_k(self):
        desired_k = 4
        elbow = Elbow(datamat=self._mat, patient_id_list=self._labels, max_k=desired_k)
        results = elbow.get_sse_for_up_to_k_clusters()
        self.assertEqual(desired_k, len(results))
