import pandas as pd
import numpy as np
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
    componentMatrix = pd.read_csv(gesture_vectors_file_path,index_col=0)
    new_newindex = []
    for ind in componentMatrix.index.to_list():
        split_ind = ind.split('_')
        new_newindex.append(split_ind[0])
    componentMatrix = pd.DataFrame(componentMatrix.values, index=new_newindex)


def initialSeedMatrix():
    global gestureMatrix
    initialseedmatrix = []
    n = len(gestureMatrix)
    for i in range(n):
        initialseedmatrix.append(1/n)
    new_index = []
    for ind in gestureMatrix.index.to_list():
        split_ind = ind.split('.')
        new_index.append(split_ind[0])
    initialseedmatrix = pd.DataFrame(initialseedmatrix, index=new_index)
    # print("initial seed matrix",initialseedmatrix.index)
    return initialseedmatrix


def getSimilarityMatrix(gestureMatrix: pd.DataFrame, initialQuery : str):
    # newSimilarityMatrix = []

    new_index = []
    for ind in gestureMatrix.index.to_list():
        split_ind = ind.split('.')
        new_index.append(split_ind[0])
    new_columns = []
    for col in gestureMatrix.columns.to_list():
        split_col = col.split('.')
        new_columns.append(split_col[0])
    newSimilarityMatrix = []
    gestureMatrix = pd.DataFrame(gestureMatrix.values, index = new_index, columns=new_columns)
    Di = gestureMatrix.loc[initialQuery]
    #print(Di)

    for i in range(len(gestureMatrix)):
        row = []
        for j, val in enumerate(Di):
            row.append(float(gestureMatrix.iloc[i,j] * val))
        newSimilarityMatrix.append(row)
    #print(newSimilarityMatrix)
    pdSimilarityMatrix = pd.DataFrame(newSimilarityMatrix, index=new_index, columns=new_columns)
    column_sums = pdSimilarityMatrix.sum(axis=1)
    nayanew_similarity_matrix = pd.DataFrame(pdSimilarityMatrix.values / column_sums.values[:, None],
                                     index=new_index, columns=new_columns)
    # print("nayanew similarity matrix",nayanew_similarity_matrix)
    return nayanew_similarity_matrix


def get_similarity_matrix(similarity_matrix_path,initialQuery):
    global gestureMatrix
    gestureMatrix = pd.read_csv(similarity_matrix_path,index_col=0)
    # print(gestureMatrix.iloc[initialQuery].shape)
    # Di = gestureMatrix.iloc[initialQuery]
    similarityMatrix = getSimilarityMatrix(gestureMatrix, initialQuery)
    return similarityMatrix

def getResult(pagerank, m=10):
    return pagerank[:m]

def modifypagerankvector(page_rank_seed_vector,relavant,irrelavant,untouched):
    global componentMatrix
    for rel in relavant:
        componentsList = componentMatrix.loc[rel]
        toAdd = 0
        denominator = 0
        for val in componentsList:
            toAdd = toAdd + val
            denominator =denominator + val**2
        page_rank_seed_vector.loc[rel] = page_rank_seed_vector.loc[rel] + (toAdd / sqrt(denominator))
        page_rank_seed_vector.loc[rel] = page_rank_seed_vector.loc[rel] / len(relavant)
    for irrel in irrelavant:
        componentsList = componentMatrix.loc[irrel]
        toSub = 0
        denominator = 0
        for val in componentsList:
            toSub = toSub + abs(val)
            denominator =denominator + val**2
        page_rank_seed_vector.loc[irrel] = page_rank_seed_vector.loc[irrel] - (toSub / sqrt(denominator))
        page_rank_seed_vector.loc[irrel] = page_rank_seed_vector.loc[irrel] / len(irrelavant)
    # print("pagerankseed vector",page_rank_seed_vector)
    # print(page_rank_seed_vector)
    # page_rank_seed_vector.to_csv("seedVector.csv")
    return page_rank_seed_vector


def get_modified_results_after_ppr(query_gesture_name: str, relevant_results: List, irrelevant_results: List, untagged_results: List,
                                                      gesture_vectors_file_path=GESTURE_VECTORS_FILE_PATH,
                                   similarity_matrix_path = SIMILARITY_MATRIX_FILE_PATH, numberOfResults = 10):
    initComponentMatrix(gesture_vectors_file_path)
    similarity_matrix = get_similarity_matrix(similarity_matrix_path,query_gesture_name)
    page_rank_seed_vector = initialSeedMatrix()
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
    relevant = args.relevant.split(',')

    if args.irrelevant is None:
        print("-i or --irrelevant argument missing")
        exit(0)
    irrelevant = args.irrelevant.split(',')

    if args.numRes is None:
        print("-t or number of results missing; using default as 10")
        numberOfResults = 10
    else:
        numberOfResults = int(args.numRes)

    new_suggestions = get_modified_results_after_ppr(query_gesture_name=query,
                                                     relevant_results=relevant,
                                                     irrelevant_results=irrelevant,
                                                     untagged_results=[],
                                                     gesture_vectors_file_path=GESTURE_VECTORS_FILE_PATH,
                                                     similarity_matrix_path = SIMILARITY_MATRIX_FILE_PATH,
                                                     numberOfResults = numberOfResults)

    print(new_suggestions)




