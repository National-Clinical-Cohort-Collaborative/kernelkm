import numpy as np


class MyMatrix:

    def __init__(self, matrix):
        if not isinstance(matrix, np.ndarray):
            raise ValueError("Can only be called on np.matrix")
        self._matrix = matrix

    def get_permuted_matrix(self) -> np.ndarray:
        """Permutate all values in matrix that are NOT on the diagonal
        Note that the matrix returned by this method is not symmetric about the diagonal
        """
        shape = self._matrix.shape
        if shape[0] != shape[1]:
            raise ValueError(f"Matrix is not a symmetrix matrix, it is {shape[0]}x{shape[1]}")
        A = np.copy(self._matrix)

        dg_idx = np.diag_indices(A.shape[0])
        A[dg_idx] = np.zeros(len(A.diagonal()))  # zero out diagonal
        idx = np.flatnonzero(A)
        A.flat[idx] = A.flat[np.random.permutation(idx)]
        A[dg_idx] = np.diagonal(self._matrix)  # replace diagonal

        return A

    def get_permuted_symmetric(self) -> np.array:
        A = np.copy(self._matrix)
        n = A.shape[0]
        upper_triangle_len = sum([i for i in range(n-1)])
        vec = np.zeros(upper_triangle_len)
        idx = 0
        for i in range(1, n):
            for j in range(i+1, n):
                elem = A[i,j]
                vec[idx] = elem
                idx += 1
        vec = vec[np.random.permutation(upper_triangle_len)]
        idx = 0
        for i in range(1,n):
            for j in range(i+1, n):
                A[i, j] = vec[idx]
                A[j, i] = vec[idx]
                idx += 1
        return A

