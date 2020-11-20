import argparse
import csv
import os
import pprint
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

similarity_matrix = pd.DataFrame.empty
component_dict = {}
gesture_subset = []
page_rank_seed_vector = []

verbose = False
pp = pprint.PrettyPrinter(indent=4)


def build_similarity_matrix():
    global similarity_matrix
    similarity_matrix = pd.read_csv('lda_similarity.csv', index_col=0, header=0)
    column_sums = similarity_matrix.sum(axis=1)
    similarity_matrix = pd.DataFrame(similarity_matrix.values / column_sums.values[:, None],
                                     index=similarity_matrix.index, columns=similarity_matrix.columns)
    if verbose:
        print("similarity matrix")
        print(similarity_matrix)


def build_graph(k=3):
    # Keep only k most similar gestures
    global similarity_matrix
    similarity_matrix = pd.DataFrame(
        np.where(similarity_matrix.rank(axis=0, method='min', ascending=False) > k, 0, similarity_matrix),
        columns=similarity_matrix.columns, index=similarity_matrix.index)

    if verbose:
        print("similarity matrix k-nearest neighbors graph")
        print(similarity_matrix)


def page_rank(matrix, num_iterations: int = 100, d: float = 0.85):
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


def find_dominant_gestures(page_rank_scores, m=6):
    return page_rank_scores[:m]


def build_component_dict(data_dir='3_class_gesture_data'):
    global component_dict
    components = ['W', 'X', 'Y', 'Z']
    for component in components:
        component_files = os.listdir(os.path.join(data_dir, component))
        for component_file in component_files:
            if component_file.lower().endswith('.csv'):
                component_file_split = component_file.split('.')
                gesture_name = component_file_split[0]
                if gesture_name not in component_dict:
                    component_dict[gesture_name] = {}
                if component not in component_dict[gesture_name]:
                    component_dict[gesture_name][component] = {}
                with open(os.path.join(data_dir, component, component_file), 'r', newline='') as read_obj:
                    csv_reader = csv.reader(read_obj)
                    for sensor_number, sensor_row in enumerate(csv_reader):
                        if sensor_number not in component_dict[gesture_name][component]:
                            component_dict[gesture_name][component][sensor_number] = []
                        row_data = []
                        for data in sensor_row:
                            row_data.append(eval(data))
                        component_dict[gesture_name][component][sensor_number] = row_data

    # if verbose:
    #     pp.pprint(component_dict)


def plot_dominant_gestures(dominant_gestures):
    if verbose:
        print(dominant_gestures)
    # style
    plt.style.use('seaborn-darkgrid')

    # create a color palette
    palette = plt.get_cmap('tab20')

    components = ['W', 'X', 'Y', 'Z']
    component_subplot_dict = {'W': (0, 0), 'X': (0, 1), 'Y': (1, 0), 'Z': (1, 1)}

    # multiple line plot
    for index, value in dominant_gestures.items():
        fig, axs = plt.subplots(2, 2)
        fig.suptitle('Gesture ' + index)
        for component in components:
            if component in component_dict[index]:
                axs[component_subplot_dict[component][0], component_subplot_dict[component][1]].set_title(component)
                num = 0
                for sensor in component_dict[index][component]:
                    axs[component_subplot_dict[component][0], component_subplot_dict[component][1]].plot(
                        component_dict[index][component][sensor],
                        marker='', color=palette(num),
                        linewidth=2, alpha=0.9, label=(str(sensor)))
                    num += 1
        plt.legend(loc='center left', bbox_to_anchor=(1, 1))
        for ax in axs.flat:
            ax.label_outer()
        plt.show()


def get_subset_from_list(gesture_set_list):
    global gesture_subset
    gesture_set_split = gesture_set_list.split(",")
    for gesture in gesture_set_split:
        gesture_subset.append(gesture.strip())


def get_subset_from_file(gesture_set_list):
    global gesture_subset
    with open(os.path.join(gesture_set_list), 'r', newline='') as read_obj:
        csv_reader = csv.reader(read_obj)
        for sensor_number, sensor_row in enumerate(csv_reader):
            for data in sensor_row:
                gesture_subset.append(data.strip())


def make_page_rank_seed_vector():
    global page_rank_seed_vector
    n = len(gesture_subset)
    for index, value in similarity_matrix.iteritems():
        if index in gesture_subset:
            page_rank_seed_vector.append(1/n)
        else:
            page_rank_seed_vector.append(0)
    page_rank_seed_vector = np.asarray(page_rank_seed_vector)


parser = argparse.ArgumentParser()
parser.add_argument("-k", "--k", help="K for k-nearest neighbors graph.")
parser.add_argument("-gesture_set_file", "--gesture_set_file", help="File listing gesture subset.")
parser.add_argument("-gesture_set_list", "--gesture_set_list",
                    help="Comma-separated list of gestures. No file "
                         "extensions.")
parser.add_argument("-m", "--m", help="# of dominant gestures to find")
parser.add_argument("-verbose", "--verbose", help="debug printing")
args = parser.parse_args()

if args.verbose is not None:
    verbose = bool(args.verbose)

if args.k is None:
    print("k for task 1 argument missing")
    exit(0)
k_from_args = int(args.k)

if args.m is None:
    print("m for task 1 argument missing")
    exit(0)
m_from_args = int(args.m)


if args.gesture_set_file is None and args.gesture_set_list is None:
    print("set set is missing. Please specify either the gesture_set_file or gesture_set_list")
    exit(0)
elif args.gesture_set_list is not None:
    get_subset_from_list(args.gesture_set_list)
else:
    get_subset_from_file(args.gesture_set_list)

build_similarity_matrix()

make_page_rank_seed_vector()

build_component_dict()

build_graph(k=k_from_args)
page_rank_values = page_rank(similarity_matrix, 100, 0.85)
dominant_gestures = find_dominant_gestures(page_rank_values, m=m_from_args)
print(dominant_gestures)
plot_dominant_gestures(dominant_gestures=dominant_gestures)
