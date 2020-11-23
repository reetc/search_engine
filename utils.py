import pandas as pd
import numpy as np


def page_rank(matrix, page_rank_seed_vector, num_iterations: int = 100, d: float = 0.85, ):
    matrix_numpy = matrix.to_numpy()

    # Get size of values
    array_size = matrix_numpy.shape[0]

    # Initialize U
    u = page_rank_seed_vector

    # Loop until convergence or num_iterations
    counter = 0
    while counter < num_iterations:
        u0 = ((1 - d) * matrix_numpy @ u) + (d * page_rank_seed_vector)
        if np.array_equal(u, u0):
            break
        u = u0
        counter += 1

    #  U to sorted pandas series
    u = pd.Series(u.flatten(), index=matrix.columns)
    u = u.sort_values(ascending=False)
    return u