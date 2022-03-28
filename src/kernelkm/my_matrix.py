
import numpy as np



class MyMatrix:

    def __init__(self, matrix):
        if not isinstance(matrix, np.matrix):
            raise ValueError("Can only be called on np.matrix")
        self._matrix = matrix



    def get_permuted_matrix(self):
        shpe = self._matrix.shape
        if shpe[0] != shpe[1]:
            raise ValueError(f"Matrix is not a symmetrix matrix, it is {shpe[0]}x{shpe[1]}")
        A = self._matrix
        m = ~np.eye(len(A), dtype=bool) # mask of non-diagonal elements
        # Extract non-diagonal elements as a new array and shuffle in-place
        Am = A[m]
        np.random.shuffle(Am)

        # Assign back the shuffled values into non-diag positions of input
        A[m] = Am