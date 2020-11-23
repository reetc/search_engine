import argparse
import random

import numpy as np
import pandas as pd

similarity_matrix = pd.DataFrame.empty


def build_similarity_matrix():
    # Fake data for now
    global similarity_matrix
    similarity_matrix = pd.read_csv('PCA_50_similarity_matrix.csv', index_col=0)
    similarity_matrix = similarity_matrix.div(similarity_matrix.sum(axis=1), axis=1)


def build_graph(k=3):
    # Keep only k most similar gestures
    global similarity_matrix
    similarity_matrix = pd.DataFrame(
        np.where(similarity_matrix.rank(axis=0, method='min', ascending=False) > k, 0, similarity_matrix),
        columns=similarity_matrix.columns, index=similarity_matrix.index)


def page_rank(matrix, num_iterations: int = 100, d: float = 0.85, n=3):
    matrix_numpy = matrix.to_numpy()

    # Get size of values
    array_size = matrix_numpy.shape[0]

    # Initialize the personalization vector to 0s
    personalize = np.zeros((array_size, 1))

    # Make randomly sampled list of n nodes
    desired_nodes = random.sample(range(array_size), n)

    # For the selected nodes, set them equal to 1/n
    for desired_node in desired_nodes:
        personalize[desired_node] = 1 / n

    # Initialize U
    u = personalize

    # Loop until convergence or num_iterations
    counter = 0
    while counter < num_iterations:
        u0 = ((1 - d) * matrix_numpy @ u) + (d * personalize)
        if np.array_equal(u, u0):
            break
        u = u0
        counter += 1

    #  U to sorted pandas series
    u = pd.Series(u.flatten(), index=matrix.columns)
    u = u.sort_values(ascending=False)
    return u


def find_dominant_gestures(page_rank_scores, m=6):
    return page_rank_scores[:6]


parser = argparse.ArgumentParser()
parser.add_argument("-k", "--k", help="K for k-nearest neighbors graph.")
parser.add_argument("-n", "--n", help="Unsure of data type")
parser.add_argument("-m", "--m", help="# of dominant gestures to find")
args = parser.parse_args()

if args.k is None:
    print("k for task 1 argument missing")
    exit(0)
k_from_args = int(args.k)

if args.n is None:
    print("n for task 1 argument missing")
    exit(0)
n_from_args = int(args.n)

if args.m is None:
    print("m for task 1 argument missing")
    exit(0)
m_from_args = int(args.m)

build_similarity_matrix()
build_graph(k=k_from_args)
page_rank_values = page_rank(similarity_matrix, 100, 0.85, n=n_from_args)
dominant_gestures = find_dominant_gestures(page_rank_values, m=m_from_args)
print(dominant_gestures)
