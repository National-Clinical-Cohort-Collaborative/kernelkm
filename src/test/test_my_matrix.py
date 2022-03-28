from unittest import TestCase
from kernelkm import MyMatrix
import numpy as np


class TestMyMatrix(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # Make 6x6 matrix
        cls._mat = np.array([[10, 5, 7, 1, 1, 1],  # Patient 1 similarities
                             [5, 10, 4, 1, 1, 1],  # Patient 2 similarities
                             [7, 4, 10, 1, 1, 1],  # Patient 3 similarities
                             [1, 1, 1, 10, 5, 5],  # Patient 4 similarities
                             [1, 1, 1, 5, 10, 5],  # Patient 5 similarities
                             [1, 1, 1, 5, 5, 10]])  # Patient 6 similarities

    def test_get_permuted_matrix(self):
        m = MyMatrix(self._mat)
        m2 = m.get_permuted_matrix()
        self.assertTrue(np.all(np.diagonal(m._matrix) == np.diagonal(m2)),
                        "diagonal should be preserved when permuting matrix")
        self.assertCountEqual(list(sorted(m._matrix.flatten())),
                              list(sorted(m2.flatten())),
                              "Values shouldn't change after permuting")