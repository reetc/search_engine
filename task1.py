import argparse
import numpy as np
import pandas as pd
import scipy as sp
import preprocessing as preprocessing
from sklearn import preprocessing

graph = []
similarity_matrix = None


def build_similarity_matrix():
    print("build_similarity_matrix")
    global similarity_matrix
    similarity_matrix = pd.read_csv('fake_similarity_matrix.csv', index_col=0)
    similarity_matrix = similarity_matrix.astype(float)
    print(similarity_matrix)


def build_graph(k=3):
    print("build_graph k={}".format(k))
    global graph

    # iterate of columns
    for (column_name, column_data) in similarity_matrix.iteritems():
        column_data = column_data.sort_values(ascending=False)
        column_data = column_data[1:k + 1]

        # iterate over items in the column
        for (item_index, item_value) in column_data.iteritems():
            # add vertex from column_name to item_index
            graph.append((column_name, item_index, item_value))
    print(graph)


def find_dominant_gestures(n=5, m=6):
    print("find_dominant_gestures n={} m={}".format(n, m))


def pagerank(M, num_iterations: int = 100, d: float = 0.85):
    desired_nodes = [0, 3, 9]
    M_numpy = M.to_numpy()
    """PageRank: The trillion dollar algorithm.

    Parameters
    ----------
    M : numpy array
        adjacency matrix where M_i,j represents the link from 'j' to 'i', such that for all 'j'
        sum(i, M_i,j) = 1
    num_iterations : int, optional
        number of iterations, by default 100
    d : float, optional
        damping factor, by default 0.85

    Returns
    -------
    numpy array
        a vector of ranks such that v_i is the i-th rank from [0, 1],
        v sums to 1

    """
    print("\n\n\n\n\n\n\n\n")
    print(M_numpy)
    print("\n\n\n\n\n\n\n\n")

    shape = M_numpy.shape
    personalize = np.zeros(shape)
    # personalize = personalize.reshape(n, 1)
    print(personalize)
    print("\n\n\n\n\n\n\n\n")

    for desired_node in desired_nodes:
        personalize[desired_node][desired_node] = 1

    print(personalize)
    print("\n\n\n\n\n\n\n\n")

    personalized_M = M_numpy @ personalize
    u = personalize

    print(personalized_M)
    print("\n\n\n\n\n\n\n\n")

    counter = 0
    while counter < num_iterations:
        u0 = ((1 - d) * personalized_M @ u) + (d * personalize)
        if (np.array_equal(u, u0)):
            break
        u = u0
        counter += 1
        # print("Counter: {}".format(counter))
        # print("u: {}".format(u))
    print("Counter: {}".format(counter))
    print(u)

    for desired_node in desired_nodes:
        # print("desired_node: {}".format(desired_node))
        n = M_numpy.shape[1]
        # print("N: {}".format(n))
        v = np.zeros((n, 1))
        v[desired_node] = 1
        # print("v: {}".format(v))
        u = v
        counter = 0
        while counter < num_iterations:
            u0 = ((1 - d) * M_numpy @ u) + (d * v)
            if (np.array_equal(u, u0)):
                break
            u = u0
            counter += 1
            # print("Counter: {}".format(counter))
            # print("u: {}".format(u))
        print("Counter: {}".format(counter))
        print(u)


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
n_from_args = args.n

if args.m is None:
    print("m for task 1 argument missing")
    exit(0)
m_from_args = int(args.m)

M = np.array([[0, 0, 0, 0, 1],
              [0.5, 0, 0, 0, 0],
              [0.5, 0, 0, 0, 0],
              [0, 1, 0.5, 0, 0],
              [0, 0, 0.5, 1, 0]])
build_similarity_matrix()

# x = similarity_matrix.values #returns a numpy array
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)
# similarity_matrix = pd.DataFrame(x_scaled, columns=similarity_matrix.columns)
similarity_matrix = similarity_matrix.div(similarity_matrix.sum(axis=1), axis=1)
print(similarity_matrix)
v_out = pagerank(similarity_matrix, 100, 0.85)

# build_graph(k=k_from_args)
# find_dominant_gestures(n=n_from_args, m=m_from_args)
