import argparse
import csv
import os
import random
from ast import literal_eval

import numpy as np
import pandas as pd

from collections import Counter

PATH_TO_TRAINING_SET = 'sample_training_labels.csv'
PATH_TO_ALL_LABELED_DATA = 'all_labels.csv'
similarity_matrix = None
labeled_data = []
all_labeled_data = []

verbose = False


def read_training_data():
    # Read the training data into a list for easy extraction
    global labeled_data
    with open(PATH_TO_TRAINING_SET, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if len(row) > 1:
                labeled_data.append((row[0], row[1]))
    if verbose:
        print("labeled_data")
        print(labeled_data)


def read_all_labeled_data():
    # Read the training data into a list for easy extraction
    global all_labeled_data
    with open(PATH_TO_ALL_LABELED_DATA, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if len(row) > 1:
                all_labeled_data.append((row[0], row[1]))
    if verbose:
        print("all_labeled_data")
        print(all_labeled_data)


def build_similarity_matrix():
    global similarity_matrix
    similarity_matrix = pd.read_csv('dtw_similarity.csv', index_col=0, header=0)
    column_sums = similarity_matrix.sum(axis=1)
    new_index = []
    for ind in similarity_matrix.index.to_list():
        split_ind = ind.split('.')
        new_index.append(split_ind[0])
    new_columns = []
    for col in similarity_matrix.columns.to_list():
        split_col = col.split('.')
        new_columns.append(split_col[0])
    similarity_matrix = pd.DataFrame(similarity_matrix.values / column_sums.values[:, None],
                                     index=new_index, columns=new_columns)
    if verbose:
        print("similarity matrix")
        print(similarity_matrix)


def find_k_nearest_neighbors(k=5):
    global similarity_matrix
    filtered_dataframe = pd.DataFrame(0.0, columns=similarity_matrix.columns, index=similarity_matrix.index)
    if verbose:
        print("filtered_dataframe")
        print(filtered_dataframe)

    for index, content in similarity_matrix.iterrows():
        label = find_training_label(index)
        if label is not None:
            filtered_dataframe.loc[index] = content
    if verbose:
        print("filtered_dataframe")
        print(filtered_dataframe)

    # For each column, retain only the K largest values
    similarity_matrix = pd.DataFrame(
        np.where(filtered_dataframe.rank(axis=0, method='min', ascending=False) > k, 0, filtered_dataframe),
        columns=filtered_dataframe.columns, index=filtered_dataframe.index)
    if verbose:
        print("similarity matrix")
        print(similarity_matrix)


def find_training_label(gesture_name):
    for labeled_datum in labeled_data:
        if labeled_datum[0] == str(gesture_name):
            return labeled_datum[1]


def k_nearest_neighbor_classifier():
    output = []

    # Loop through each gesture in the dataset
    for label, content in similarity_matrix.iteritems():
        labels = []

        # Loop through each value in the series
        for index, value in content.items():

            # If the value is not zero (meaning it is one of the K largest values) add its training label to the list
            if value != 0:
                labels.append(find_training_label(index))
                # if verbose:
                #     print("value")
                #     print(value)
                #     print("training label")
                #     print(find_training_label(index))
        c = Counter(labels)

        # Use the most common label as the found label for this gesture
        most_common = c.most_common(n=1)[0][0]
        output.append((label, most_common))
    return output


def page_rank_classifier(num_iterations: int = 100, d: float = 0.85):
    matrix_numpy = similarity_matrix.to_numpy()

    output = []

    # Get size of values
    array_size = matrix_numpy.shape[0]
    for label, content in similarity_matrix.iteritems():
        print(label)
        index_to_classify = similarity_matrix.columns.get_loc(label)

        # Initialize the personalization vector to 0s
        personalize = np.zeros((array_size, 1))

        # For the selected nodes, set them equal to 1/n
        personalize[index_to_classify] = 1

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
        u = pd.Series(u.flatten(), index=similarity_matrix.columns)
        u = u.sort_values(ascending=False)
        print(u.index[1])
        output.append((label, find_training_label(u.index[1])))
        print()
    return output


def build_frequency_dataset(frequency_definition):
    vector_path = os.path.join('vectors')
    file_names = os.listdir(vector_path)
    files_to_use = []
    for file_name in file_names:
        if file_name.split('.')[0] == ('vector_'+frequency_definition):
            files_to_use.append(file_name)
    for file_to_use in files_to_use:
        with open(os.path.join('vector', file_to_use)) as f:
            object_vector = [list(literal_eval(line)) for line in f]
        final_vector = []
        for i, row in enumerate(object_vector):
            if i % 2 != 0:
                final_vector += row


def get_gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini


def decision_tree_classifier(frequency_definition):



def find_accuracy(found_labels):
    num_correct = 0
    for found_label in found_labels:
        if found_label in all_labeled_data:
            num_correct += 1
    return num_correct/len(all_labeled_data)


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--c",
                    help="Classifier. Options: 1. K-Nearest Neighbor 2. Personalized-PageRank 3. Decision Tree")
parser.add_argument("-k", "--k", help="K for k-nearest neighbors classifier.")
parser.add_argument("-verbose", "--verbose", help="debug printing")

args = parser.parse_args()

if args.verbose is not None:
    verbose = bool(args.verbose)

if args.c is None:
    print("c for task 2 argument missing")
    exit(0)
else:
    if args.c not in ['1', '2', '3']:
        print("Unrecognized Classifier")
        exit(0)
c_from_args = args.c

read_training_data()
read_all_labeled_data()

if c_from_args in ['1', '2']:
    if args.k is None:
        print("k for task 1 argument missing")
        exit(0)
    k_from_args = int(args.k)
    build_similarity_matrix()
    find_k_nearest_neighbors(k=k_from_args)

if c_from_args == '1':
    found_labels = k_nearest_neighbor_classifier()
elif c_from_args == '2':
    found_labels = page_rank_classifier()
elif c_from_args == '3':
    found_labels = decision_tree_classifier()

for found_label in found_labels:
    print(found_label)

accuracy = find_accuracy(found_labels)
print(accuracy)
