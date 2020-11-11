import argparse
import csv
import random

import numpy as np
import pandas as pd

from collections import Counter

PATH_TO_TRAINING_SET = 'fake_sample_training_labels.csv'
similarity_matrix = None
labeled_data = []


def read_training_data():
    # Read the training data into a list for easy extraction
    global labeled_data
    with open(PATH_TO_TRAINING_SET, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if len(row) > 1:
                labeled_data.append((row[0], row[1]))


def build_similarity_matrix():
    global similarity_matrix

    # Fake data for now
    similarity_matrix = pd.read_csv('fake_similarity_matrix.csv', index_col=0)

    # Normalize the data column-wise
    similarity_matrix = similarity_matrix.div(similarity_matrix.sum(axis=1), axis=1)


def find_k_nearest_neighbors(k=5):
    # For each column, retain only the K largest values
    global similarity_matrix
    similarity_matrix = pd.DataFrame(
        np.where(similarity_matrix.rank(axis=0, method='min', ascending=False) > k, 0, similarity_matrix),
        columns=similarity_matrix.columns, index=similarity_matrix.index)


def find_training_label(gesture_name):
    for labeled_datum in labeled_data:
        if labeled_datum[0] == gesture_name:
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


def decision_tree_classifier():
    pass


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--c",
                    help="Classifier. Options: 1. K-Nearest Neighbor 2. Personalized-PageRank 3. Decision Tree")
parser.add_argument("-k", "--k", help="K for k-nearest neighbors classifier.")

args = parser.parse_args()

if args.c is None:
    print("c for task 2 argument missing")
    exit(0)
else:
    if args.c not in ['1', '2', '3']:
        print("Unrecognized Classifier")
        exit(0)
c_from_args = args.c

if c_from_args in ['1', '2']:
    if args.k is None:
        print("k for task 1 argument missing")
        exit(0)
    k_from_args = int(args.k)
    build_similarity_matrix()
    find_k_nearest_neighbors(k=k_from_args)

read_training_data()

if c_from_args == '1':
    found_labels = k_nearest_neighbor_classifier()
elif c_from_args == '2':
    found_labels = page_rank_classifier()
elif c_from_args == '3':
    found_labels = decision_tree_classifier()

for found_label in found_labels:
    print(found_label)
