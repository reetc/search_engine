import pandas as pd
import numpy as np
import os
from utils import *
from math import sqrt
import argparse
from typing import List

gestureMatrix = pd.DataFrame.empty
componentMatrix = pd.DataFrame.empty
GESTURE_VECTORS_FILE_PATH = "latent_features.csv"
SIMILARITY_MATRIX_FILE_PATH = "SVD_50_similarity_matrix.csv"


def initComponentMatrix(gesture_vectors_file_path):
    global componentMatrix
    componentMatrix = pd.read_csv(gesture_vectors_file_path, index_col=0)
    new_newindex = []
    for ind in componentMatrix.index.to_list():
        split_ind = ind.split('_')
        new_newindex.append(split_ind[0])
    componentMatrix = pd.DataFrame(componentMatrix.values, index=new_newindex)


def initialSeedMatrix(similarityMatrix):
    # global gestureMatrix
    # gestureMatrix = pd.read_csv(gesture_vectors_file_path, index_col=0)
    initialseedmatrix = []
    n = len(similarityMatrix)
    for i in range(n):
        initialseedmatrix.append(1 / n)
    new_index = []
    for ind in similarityMatrix.index.to_list():
        split_ind = ind.split('_')
        new_index.append(split_ind[0])
    initialseedmatrix = pd.DataFrame(initialseedmatrix, index=new_index)
    # print("initial seed matrix",initialseedmatrix.index)
    return initialseedmatrix


def getSimilarityMatrix(gestureMatrix: pd.DataFrame, initialQuery: str):
    # tempSimilarityMatrix = []

    new_index = []
    # print(len(gestureMatrix))
    for ind in gestureMatrix.index.to_list():
        split_ind = ind.split('_')
        new_index.append(split_ind[0])
    new_columns = []
    for col in gestureMatrix.columns.to_list():
        split_col = col.split('_')
        new_columns.append(split_col[0])
    tempSimilarityMatrix = []
    gestureMatrix = pd.DataFrame(gestureMatrix.values, index=new_index, columns=new_columns)
    Di = gestureMatrix.loc[initialQuery]
    # print(Di)

    for i in range(len(gestureMatrix)):
        row = []
        for j, val in enumerate(Di):
            row.append(float(gestureMatrix.iloc[i, j] * val))
        tempSimilarityMatrix.append(row)
    # print(tempSimilarityMatrix)
    pdSimilarityMatrix = pd.DataFrame(tempSimilarityMatrix, index=new_index, columns=new_columns)
    column_sums = pdSimilarityMatrix.sum(axis=1)
    similarityMatrix = pd.DataFrame(pdSimilarityMatrix.values / column_sums.values[:, None],
                                    index=new_index, columns=new_columns)
    # print("nayanew similarity matrix",similarityMatrix)
    return similarityMatrix


def get_similarity_matrix(similarity_matrix_path, initialQuery):
    global gestureMatrix
    gestureMatrix = pd.read_csv(similarity_matrix_path, index_col=0)
    # print(gestureMatrix.iloc[initialQuery].shape)
    # Di = gestureMatrix.iloc[initialQuery]
    similarityMatrix = getSimilarityMatrix(gestureMatrix, initialQuery)
    return similarityMatrix


def getResult(pagerank, m=10):
    return pagerank[:m]


def modifypagerankvector(page_rank_seed_vector, relavant=[], irrelavant=[], untouched=[]):
    global componentMatrix
    # print(relavant)
    # print(irrelavant)
    if relavant:
        for rel in relavant:
            componentsList = componentMatrix.loc[rel]
            toAdd = 0
            denominator = 0
            for val in componentsList:
                toAdd = toAdd + val
                denominator = denominator + val ** 2
            page_rank_seed_vector.loc[rel] = page_rank_seed_vector.loc[rel] + (toAdd / sqrt(denominator))
            page_rank_seed_vector.loc[rel] = page_rank_seed_vector.loc[rel] / len(relavant)
    if irrelavant:
        for irrel in irrelavant:
            componentsList = componentMatrix.loc[irrel]
            toSub = 0
            denominator = 0
            for val in componentsList:
                toSub = toSub + abs(val)
                denominator = denominator + val ** 2
            page_rank_seed_vector.loc[irrel] = page_rank_seed_vector.loc[irrel] - (toSub / sqrt(denominator))
            page_rank_seed_vector.loc[irrel] = page_rank_seed_vector.loc[irrel] / len(irrelavant)
    # print("pagerankseed vector",page_rank_seed_vector)
    # print(page_rank_seed_vector)
    # page_rank_seed_vector.to_csv("seedVector.csv")
    # print(type(page_rank_seed_vector))
    return page_rank_seed_vector


def build_graph(similarity_matrix, connected_nodes=15):
    # Keep only k most similar gestures
    # global similarity_matrix
    similarity_matrix = pd.DataFrame(
        np.where(similarity_matrix.rank(axis=0, method='min', ascending=False) > connected_nodes, 0, similarity_matrix),
        columns=similarity_matrix.columns, index=similarity_matrix.index)
    return similarity_matrix


def get_modified_results_after_ppr(query_gesture_name: str,
                                   relevant_results: List,
                                   irrelevant_results: List,
                                   untagged_results: List,
                                   gesture_vectors_file_path=GESTURE_VECTORS_FILE_PATH,
                                   similarity_matrix_path=SIMILARITY_MATRIX_FILE_PATH,
                                   numberOfResults=10,
                                   connected_nodes=15):
    initComponentMatrix(gesture_vectors_file_path)
    if not os.path.exists('similarity_matrices'):
        os.makedirs('similarity_matrices')
    similarityFile = "SimilarityMatrix_" + query_gesture_name + "_" + str(connected_nodes)
    pathSM = os.path.abspath("similarity_matrices/" + similarityFile + ".csv")
    if (os.path.exists(pathSM)):
        similarity_matrix = pd.read_csv(pathSM, index_col=0)
    else:
        similarity_matrix = get_similarity_matrix(similarity_matrix_path, query_gesture_name)
        similarity_matrix = build_graph(similarity_matrix, connected_nodes)
        similarity_matrix.to_csv(pathSM)

    # print(similarity_matrix)
    page_rank_seed_vector = initialSeedMatrix(similarity_matrix)
    page_rank_seed_vector = modifypagerankvector(page_rank_seed_vector, relevant_results,
                                                 irrelevant_results, untagged_results)
    page_rank_seed_vector = np.asarray(page_rank_seed_vector)
    page_rank_values = page_rank(similarity_matrix, page_rank_seed_vector, 100, 0.85)
    # page_rank_values.to_csv("RankValues.csv")
    result = getResult(page_rank_values, numberOfResults)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", help="list of relevant gestures, e.g. --query=570")
    parser.add_argument("-t", "--numRes", help="list of relevant information results")
    parser.add_argument("-k", "--connections", help="no of nodes each node is connected to")
    parser.add_argument("-r", "--relevant", help="list of relevant gestures, e.g. --relevant=570,570,571,574,575")
    parser.add_argument("-i", "--irrelevant", help="list of irrelevant gestures, e.g. --irrelevant=572,573")
    args = parser.parse_args()

    if args.query is None:
        print("-q or --query argument missing")
        exit(0)
    query = args.query

    if args.relevant is None:
        print("-r or --relevant argument missing")
        exit(0)
    elif args.relevant == "":
        relevant = []
    else:
        relevant = args.relevant.split(',')

    if args.irrelevant is None:
        print("-i or --irrelevant argument missing")
        exit(0)
    elif args.irrelevant == "":
        irrelevant = []
    else:
        irrelevant = args.irrelevant.split(',')

    if args.numRes is None:
        print("-t or number of results missing; using default as 10")
        numberOfResults = 10
    else:
        numberOfResults = int(args.numRes)

    if args.connections is None:
        print("-k or number of nodes to connect is missing; using default as 15")
        connections = 15
    else:
        connections = int(args.connections)

    new_suggestions = get_modified_results_after_ppr(query_gesture_name=query,
                                                     relevant_results=relevant,
                                                     irrelevant_results=irrelevant,
                                                     untagged_results=[],
                                                     gesture_vectors_file_path=GESTURE_VECTORS_FILE_PATH,
                                                     similarity_matrix_path=SIMILARITY_MATRIX_FILE_PATH,
                                                     numberOfResults=numberOfResults,
                                                     connected_nodes=connections)

    print("--------------------------------")
    print("Gesture Name\t", "Similarity")
    print("--------------------------------")
    for gesture_name, similarity in new_suggestions.iteritems():
        print(str(gesture_name) + "\t\t", str(similarity))