import pandas as pd
import numpy as np
from utils import *
import math
gestureMatrix = pd.DataFrame.empty
componentMatrix = pd.DataFrame.empty


def initComponentMatrix():
    global componentMatrix
    componentMatrix = pd.read_csv("latent_features.csv",index_col=0)
    new_newindex = []
    for ind in componentMatrix.index.to_list():
        split_ind = ind.split('_')
        # val = split_ind[0]
        new_newindex.append(split_ind[0])
    componentMatrix = pd.DataFrame(componentMatrix.values, index=new_newindex)
    print(componentMatrix)
    # print("testing component", componentMatrix.loc["1-1"])

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
    print("initial seed matrix",initialseedmatrix.index)
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


def get_similarity_matrix(initialQuery):
    global gestureMatrix
    gestureMatrix = pd.read_csv("SVD_50_similarity_matrix.csv",index_col=0)
    # print(gestureMatrix.iloc[initialQuery].shape)
    # Di = gestureMatrix.iloc[initialQuery]
    similarityMatrix = getSimilarityMatrix(gestureMatrix, initialQuery)
    return similarityMatrix

def getResult(pagerank, m=10):
    return pagerank[:m]



def modifypagerankvector(page_rank_seed_vector,relavant,irrelavant,untouched):
    global componentMatrix

    # for index,value in componentMatrix.iteritems():
    #     if index in relavant:
    #         componentsList = componentMatrix.loc[index]
    #         rel=index
    #         toAdd = 0
    #         denominator = 0
    #         for val in componentsList:
    #             val = float(val)
    #             toAdd = toAdd + abs(val)
    #             denominator = denominator + val ** 2
    #         print("todadd, denominatio", toAdd,denominator)
    #         page_rank_seed_vector.loc[index] = float(page_rank_seed_vector.loc[index]) + (toAdd / math.sqrt(denominator))
    #         page_rank_seed_vector.loc[index] = float(page_rank_seed_vector.loc[index])/ len(relavant)
    #
    # for index,value in componentMatrix.iteritems():
    #     componentsList=[]
    #     if index in irrelavant:
    #         componentsList = componentMatrix.loc[index]
    #         toSub = 0
    #         irrel=index
    #         denominator = 0
    #         for val in componentsList:
    #             val = float(val)
    #             toSub = toSub + val
    #             denominator = denominator + val ** 2
    #         page_rank_seed_vector.loc[index] = page_rank_seed_vector.loc[index] - (toSub / math.sqrt(denominator))
    #         page_rank_seed_vector.loc[index] = page_rank_seed_vector.loc[index] / len(irrelavant)

    for rel in relavant:
        componentsList = componentMatrix.loc[rel]
        toAdd = 0
        denominator = 0
        for val in componentsList:
            toAdd = toAdd + val
            denominator =denominator + val**2
        print("toadd",toAdd)
        print("denominator",denominator)
        page_rank_seed_vector.loc[rel] = page_rank_seed_vector.loc[rel] + (toAdd / math.sqrt(denominator))
        page_rank_seed_vector.loc[rel] = page_rank_seed_vector.loc[rel] / len(relavant)
    for irrel in irrelavant:
        componentsList = componentMatrix.loc[irrel]
        toSub = 0
        denominator = 0
        for val in componentsList:
            toSub = toSub + abs(val)
            denominator =denominator + val**2
        page_rank_seed_vector.loc[irrel] = page_rank_seed_vector.loc[irrel] - (toSub / math.sqrt(denominator))
        page_rank_seed_vector.loc[irrel] = page_rank_seed_vector.loc[irrel] / len(irrelavant)
    # print("pagerankseed vector",page_rank_seed_vector)
    return page_rank_seed_vector

if __name__ == "__main__":
    initComponentMatrix()
    similarity_matrix = get_similarity_matrix("1-5")
    page_rank_seed_vector = initialSeedMatrix()


    #page_rank_seed_vector=np.asarray(page_rank_seed_vector)
    #print(page_rank_seed_vector)
    page_rank_seed_vector=modifypagerankvector(page_rank_seed_vector,["1-1","2-3","3-4","4-5"],["250-1","251-2","252-3"],[])
    # print("final seed index",page_rank_seed_vector.index)
    page_rank_seed_vector =np.asarray(page_rank_seed_vector)
    # print("final seed",page_rank_seed_vector.shape)
    # print("final seed",page_rank_seed_vector)
    # print("final simil",similarity_matrix)
    page_rank_values = page_rank(similarity_matrix, page_rank_seed_vector, 100, 0.85)
    # print("pagerankkvalues",page_rank_values)
    result = getResult(page_rank_values)
    print("result",result)

