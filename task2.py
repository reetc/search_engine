import argparse
import csv

import numpy as np
import pandas as pd

PATH_TO_TRAINING_SET = 'fake_sample_training_labels.csv'
similarity_matrix = None
labeled_data = []


def read_training_data():
    global labeled_data
    with open(PATH_TO_TRAINING_SET, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if len(row) > 1:
                labeled_data.append((row[0], row[1]))
    print(labeled_data)


def build_similarity_matrix():
    # Fake data for now
    global similarity_matrix
    similarity_matrix = pd.read_csv('fake_similarity_matrix.csv', index_col=0)
    similarity_matrix = similarity_matrix.div(similarity_matrix.sum(axis=1), axis=1)


def k_nearest_neighbor_classifier(k=5):
    global similarity_matrix
    similarity_matrix = pd.DataFrame(
        np.where(similarity_matrix.rank(axis=0, method='min', ascending=False) > k, 0, similarity_matrix),
        columns=similarity_matrix.columns, index=similarity_matrix.index)

    # Use most common label.
    pass


def page_rank_classifier():
    pass


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

read_training_data()

if c_from_args == '1':
    if args.k is None:
        print("k for task 1 argument missing")
        exit(0)
    k_from_args = int(args.k)
    k_nearest_neighbor_classifier(k=k_from_args)
elif c_from_args == '2':
    page_rank_classifier()
elif c_from_args == '3':
    decision_tree_classifier()
