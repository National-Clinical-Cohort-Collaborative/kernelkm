import numpy as np


class MyMatrix:

    def __init__(self, matrix):
        if not isinstance(matrix, np.ndarray):
            raise ValueError("Can only be called on np.matrix")
        self._matrix = matrix

    def get_permuted_matrix(self):
        shpe = self._matrix.shape
        if shpe[0] != shpe[1]:
            raise ValueError(f"Matrix is not a symmetrix matrix, it is {shpe[0]}x{shpe[1]}")
        # A = self._matrix
        A = np.copy(self._matrix)
        dg = A.diagonal()
        dg_idx = np.diag_indices(A.shape[0])
        A[dg_idx] = np.zeros(len(dg))
        idx = np.flatnonzero(A)
        A.flat[idx] = A.flat[np.random.permutation(idx)]
        # A.diagonal = dg

        return A