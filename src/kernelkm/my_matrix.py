
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
        A = self._matrix
        #m = ~np.eye(len(A), dtype=bool) # mask of non-diagonal elements
        # Extract non-diagonal elements as a new array and shuffle in-place
        #Am = A[m]
        #np.random.shuffle(Am)
        #idx = np.flatnonzero(m)
        #A.flat[idx] = A.flat[np.random.permutation(idx)]
        dg = A.diagonal()
        dg_idx = np.diag_indices(A)
        A[dg_idx] = np.zeros(len(dg))
        idx = np.flatnonzero(A)
        A.flat[idx] = A.flat[np.random.permutation(idx)]
        A.diagonal = dg

        # Assign back the shuffled values into non-diag positions of input
        #A[m] = Am
        return A