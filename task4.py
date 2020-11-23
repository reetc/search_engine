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
    gesture_vectors_df = utils.get_gesture_vectors(GESTURE_VECTORS_FILE_PATH)
    _binarize(gesture_vectors_df)
    print(gesture_vectors_df)
    similarity_gesture_pairs = []
    for gesture_index, gesture_row in gesture_vectors_df.iterrows():
        similarity = sim(gesture_row, gesture_vectors_df.loc[gesture_name], cleaned_relevant_result_names,
                         gesture_vectors_df,
                         gesture_name, gesture_index)
        similarity_gesture_pairs.append((similarity, gesture_index))
    return sorted(similarity_gesture_pairs)


def _binarize(df: DataFrame):
    column_names = df.columns
    for column in column_names:
        df[column] = _binarize_column(df[column])


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


def sim(d: list, q: list, relevant_results: List, df: DataFrame, q_gesture_name, d_gesture_name):
    R = len(relevant_results)
    N = len(df)
    n = count_of_objects_in_which_di_1(df)
    r = count_of_relevant_objects_in_which_di_1(relevant_results, df)
    sum = 0
    for i in range(len(df.columns)):
        pi = (r[i] + 0.5) / (R + 1)
        ui = (n[i] - r[i] + 0.5) / (N - R + 1)
        sum += d[i] * log(pi * (1 - ui) / ui * (1 - pi)) + np.dot(df.loc[q_gesture_name], df.loc[d_gesture_name])
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
        "1",
        # ["561", "562", "563", "563", "564", "565", "566", "567", "568"],
        # [str(x) for x in range(559, 589+1)],
        # [str(x) for x in range(270, 279+1)],
        [str(x) for x in range(1, 30 + 1)],
        [],
        []
    )
    print("Outputs:")
    pprint(similarity_gesture_pairs)
