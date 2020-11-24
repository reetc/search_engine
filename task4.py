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


def get_modified_results_after_probabilistic_feedback(query_gesture_name: str,
                                                      relevant_results: List,
                                                      irrelevant_results: List,
                                                      untagged_results: List):
    cleaned_relevant_result_names = [e.split('_')[0] for e in relevant_results]
    cleaned_irrelevant_result_names = [e.split('_')[0] for e in irrelevant_results]
    all_results = cleaned_relevant_result_names + cleaned_irrelevant_result_names
    bdf = _binarize(utils.get_gesture_vectors(GESTURE_VECTORS_FILE_PATH))
    # print(bdf)
    similarity_gesture_pairs = []
    dissimilarity_gesture_pairs = []
    global NI_COUNTS, RI_COUNTS
    for gesture_name, gesture_vector in bdf.loc[all_results].iterrows():
        similarity_to_relevant_gestures = sim(d=gesture_vector, relevant_results=cleaned_relevant_result_names, bdf=bdf)
        similarity_gesture_pairs.append((similarity_to_relevant_gestures, gesture_name))
    NI_COUNTS = []  # Reset for re-running
    RI_COUNTS = []
    for gesture_name, gesture_vector in bdf.loc[all_results].iterrows():
        similarity_to_irrelevant_gestures = sim(d=gesture_vector, relevant_results=cleaned_irrelevant_result_names,
                                                bdf=bdf)
        dissimilarity_gesture_pairs.append((similarity_to_irrelevant_gestures, gesture_name))
    NI_COUNTS = []
    RI_COUNTS = []
    final_similarity_gesture_pairs = [(sgp[0]-0.5*dgp[0], sgp[1]) for sgp,dgp in
                                      zip(similarity_gesture_pairs, dissimilarity_gesture_pairs)]
    similarity_gesture_pairs = list(set(final_similarity_gesture_pairs))  # Remove duplicates
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


def sim(d: list, relevant_results: List, bdf: DataFrame):
    N = len(bdf)
    R = len(relevant_results)
    R = min(N, R)  # Preventing negative values for (N-R)
    n = count_of_objects_in_which_di_1(bdf)
    r = count_of_relevant_objects_in_which_di_1(relevant_results, bdf)
    sum_value = 0
    for i in range(len(bdf.columns)):
        # pi = (r[i] + 0.5) / (R + 1)
        # ui = (n[i] - r[i] + 0.5) / (N - R + 1)
        pi = (r[i] + n[i]/N) / (R + 1)
        ui = (n[i] - r[i] + n[i]/N) / (N - R + 1)

        # Making error corrections
        pi = min(0.9999, pi)
        ui = min(0.9999, ui)

        num = pi*(1 - ui)
        den = ui*(1 - pi)

        sum_value += d[i] * log(num / den)
    return sum_value


def count_of_objects_in_which_di_1(df: DataFrame):
    global NI_COUNTS
    if len(NI_COUNTS) != 0:
        return NI_COUNTS
    for column_name in df.columns:
        entries_with_1 = df[df[column_name] == 1]
        NI_COUNTS.append(len(entries_with_1))
        # print(df[df[column_name] == 1].index)
        # print(df.index)
    print("NI_COUNTS", NI_COUNTS)
    return NI_COUNTS


def count_of_relevant_objects_in_which_di_1(relevant_results: List, df: DataFrame):
    # print(relevant_results)
    global RI_COUNTS
    if len(RI_COUNTS) != 0:
        return RI_COUNTS
    relevant_df = df.loc[relevant_results]
    # print(relevant_df)
    for column_name in df.columns:
        entries_with_1 = relevant_df[relevant_df[column_name] == 1]
        RI_COUNTS.append(len(entries_with_1))
    RI_COUNTS = [ri if ri<=ni else ni for ni,ri in zip(NI_COUNTS,RI_COUNTS)]
    print("RI_COUNTS", RI_COUNTS)
    return RI_COUNTS


if __name__ == "__main__":
    pd.set_option('display.max_rows', 500)
    similarity_gesture_pairs = get_modified_results_after_probabilistic_feedback(
        "570",
        # ["561", "562", "563", "563", "564", "565", "566", "567", "568", "560"],
        # [str(x) for x in range(559, 589+1)],
        # [str(x) for x in [250, 253, 258, 263, 265, 577, 582, 1]],
        # [str(x) for x in range(1, 10+1)] + ["1"]*1,
        # ["561"] * 10,
        # ["571", "572", "573", "574", "575", "576"] + ["570"]*100, ["260", "261", "262"] + ["261"]*5,
        ["571", "572", "573", "574", "575"]+["570"]*3+["574"]+["571"], ["572", "573"],
        []
    )
    print("Outputs:")
    pprint(similarity_gesture_pairs)
