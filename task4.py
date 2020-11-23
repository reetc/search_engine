from math import log
from pprint import pprint
from typing import List

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.cluster import KMeans

import utils

GESTURE_VECTORS_FILE_PATH = "latent_features.csv"
NI_COUNTS = []
RI_COUNTS = []


def get_modified_results_after_probabilistic_feedback(gesture_name: str,
                                                      relevant_results: List,
                                                      irrelevant_results: List,
                                                      untagged_results: List):
    cleaned_relevant_result_names = [e.split('_')[0] for e in relevant_results]
    df = utils.get_gesture_vectors(GESTURE_VECTORS_FILE_PATH)
    bdf = _binarize(df)
    print(df)
    similarity_gesture_pairs = []
    for gesture_index, gesture_row in df.iterrows():
        similarity = sim(d=gesture_row, q=df.loc[gesture_name],
                         relevant_results=cleaned_relevant_result_names,
                         bdf=bdf, odf=df,
                         q_gesture_name=gesture_name, d_gesture_name=gesture_index)
        similarity_gesture_pairs.append((similarity, gesture_index))
    return sorted(similarity_gesture_pairs)


def _binarize(df: DataFrame):
    copied_df = df.copy()
    column_names = copied_df.columns
    for column in column_names:
        copied_df[column] = _binarize_column(copied_df[column])
    return copied_df


def _binarize_column(s: Series):
    arr = np.array(s).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(arr)

    zero_label_mu = kmeans.cluster_centers_[0]
    one_label_mu = kmeans.cluster_centers_[1]
    is_one_label_max = abs(one_label_mu) > abs(zero_label_mu)
    should_we_invert_labels = not is_one_label_max
    labels = kmeans.labels_
    if should_we_invert_labels:
        labels = [1 if x == 0 else 0 for x in labels]
    return pd.Series(labels, name=s.name, index=s.index)


def sim(d: list, q: list, relevant_results: List, bdf: DataFrame, odf: DataFrame, q_gesture_name, d_gesture_name):
    R = len(relevant_results)
    N = len(bdf)
    n = count_of_objects_in_which_di_1(bdf)
    r = count_of_relevant_objects_in_which_di_1(relevant_results, bdf)
    sum = 0
    for i in range(len(bdf.columns)):
        pi = (r[i] + 0.5) / (R + 1)
        ui = (n[i] - r[i] + 0.5) / (N - R + 1)
        sum += d[i] * log(pi * (1 - ui) / ui * (1 - pi)) + \
               10*np.dot(odf.loc[q_gesture_name], odf.loc[d_gesture_name])
    return sum


def count_of_objects_in_which_di_1(df: DataFrame):
    global NI_COUNTS
    if len(NI_COUNTS) != 0:
        return NI_COUNTS
    for column_name in df.columns:
        entries_with_1 = df[df[column_name] == 1]
        NI_COUNTS.append(len(entries_with_1))
    print("NI_COUNTS", NI_COUNTS)
    return NI_COUNTS


def count_of_relevant_objects_in_which_di_1(relevant_results: List, df: DataFrame):
    global RI_COUNTS
    if len(RI_COUNTS) != 0:
        return RI_COUNTS
    relevant_df = df.loc[relevant_results]
    for column_name in df.columns:
        entries_with_1 = relevant_df[relevant_df[column_name] == 1]
        RI_COUNTS.append(len(entries_with_1))
    print("RI_COUNTS", RI_COUNTS)
    return RI_COUNTS


if __name__ == "__main__":
    pd.set_option('display.max_rows', 500)
    similarity_gesture_pairs = get_modified_results_after_probabilistic_feedback(
        "271",
        # ["561", "562", "563", "563", "564", "565", "566", "567", "568", "560"],
        # [str(x) for x in range(559, 589+1)],
        [str(x) for x in [250, 253, 258, 263, 265, 577, 582, 1]],
        # [str(x) for x in range(1, 30 + 1)],
        # ["561"] * 10,
        # ["571", "572", "573", "574", "575", "576"],
        [],
        []
    )
    print("Outputs:")
    pprint(similarity_gesture_pairs)
