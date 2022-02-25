import numpy as np 
from unittest import TestCase
from kernelkm import KernelKMeans



class TestKMeans(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        ## Make 6x6 matrix
        cls._mat = np.array([[0, 5, 6, 1, 1, 1], # Patient 1 similarities
                             [5, 0, 4, 1, 1, 1], # Patient 2 similarities
                             [6, 4, 0, 1, 1, 1], # Patient 3 similarities
                             [1, 1, 1, 0, 5, 5], # Patient 4 similarities
                             [1, 1, 1, 5, 0, 5], # Patient 5 similarities
                             [1, 1, 1, 5, 5, 0]]) # Patient 6 similarities
        cls._labels = ["p1","p2","p3","p4","p5","p6"]

    def test_ctor(self):
        kkm = KernelKMeans(datamat=self._mat, patient_id_list = self._labels)
        self.assertIsNotNone(kkm)

    def test_clustering(self):
        kkm = KernelKMeans(datamat=self._mat, patient_id_list = self._labels)
        kkm.calculate(k=4)






