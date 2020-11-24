import argparse
import csv

import numpy as np
import pandas as pd

from collections import Counter

from utils import page_rank

PATH_TO_TRAINING_SET = 'sample_training_labels.csv'
PATH_TO_ALL_LABELED_DATA = 'all_labels.csv'
similarity_matrix = None
labeled_data = []
all_labeled_data = []
dt_file_order = []

verbose = False


def read_training_data():
    # Read the training data into a list for easy extraction
    global labeled_data
    with open(PATH_TO_TRAINING_SET, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
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
    similarity_matrix = pd.read_csv('SVD_50_similarity_matrix.csv', index_col=0, header=0)
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
    # noinspection PyTypeChecker
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


def find_real_label(gesture_name):
    for labeled_datum in all_labeled_data:
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
        # print(label)
        index_to_classify = similarity_matrix.columns.get_loc(label)

        # Initialize the personalization vector to 0s
        personalize = np.zeros((array_size, 1))

        # For the selected nodes, set them equal to 1/n
        personalize[index_to_classify] = 1

        u = page_rank(similarity_matrix, personalize, 100, 0.85)
        # print(u.index[1])
        output.append((label, find_training_label(u.index[1])))
        # print()
    return output


# Currently GINI index
def get_score(groups, classes):
    # count all samples at split point
    n_instances = 0.0
    for group in groups:
        n_instances += float(len(group))
    # sum weighted Gini index for each group
    output = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            proportion = 0
            for row in group:
                if row[-1] == class_val:
                    proportion += 1
            proportion = proportion / size
            score += proportion * proportion
        # weight the group score by its relative size
        output += (1.0 - score) * (size / n_instances)
    return output


# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Create a terminal node value
def to_terminal(group):
    outcomes = []
    for row in group:
        outcomes.append(row[-1])
    return max(set(outcomes), key=outcomes.count)


# Select the best split point for a dataset
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    selected_index, selected_value, selected_score, selected_groups = 999, 999, 999, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            score = get_score(groups, class_values)
            if score < selected_score:
                selected_index, selected_value, selected_score, selected_groups = index, row[index], score, groups
    return {'index': selected_index, 'value': selected_value, 'groups': selected_groups}


# Create child splits for a node or make terminal
def split(node, max_depth, depth, max_num):
    left, right = node['groups']
    del (node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return

    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return

    # process left child
    if len(left) <= max_num:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, depth + 1, max_num)

    # process right child
    if len(right) <= max_num:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, depth + 1, max_num)


# Build a decision tree
def build_tree(train, max_depth, max_num):
    root = get_split(train)
    split(root, max_depth, 1, max_num)
    return root


def get_file_order(frequency_definition):
    global dt_file_order
    with open('file_order_' + frequency_definition + '.txt', 'r', newline='') as order_file:
        for index, line in enumerate(order_file):
            dt_file_order.append(line)


def get_frequency_training_set(frequency_definition):
    dataset = []

    with open('dataset_' + frequency_definition + '.txt', 'r', newline='') as csv_file:
        for index, row in enumerate(csv_file):
            if find_training_label(dt_file_order[index].split('_')[0]) is not None:
                data_row = eval(row)
                data_row.append(find_training_label(dt_file_order[index].split('_')[0]))
                dataset.append(data_row)
    # print(dataset)
    return dataset


def get_full_frequency_set(frequency_definition):
    dataset = []

    with open('dataset_' + frequency_definition + '.txt', 'r', newline='') as csv_file:
        for index, row in enumerate(csv_file):
            if find_real_label(dt_file_order[index].split('_')[0]) is not None:
                data_row = eval(row)
                data_row.append(find_real_label(dt_file_order[index].split('_')[0]))
                dataset.append(data_row)
    # print(dataset)
    return dataset


# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


def decision_tree_classifier(frequency_definition):
    output = []
    get_file_order(frequency_definition)
    training_dataset = get_frequency_training_set(frequency_definition)
    decision_tree = build_tree(training_dataset, 80, 3)
    full_dataset = get_full_frequency_set(frequency_definition)
    #  predict with a stump
    for index, row in enumerate(full_dataset):
        prediction = predict(decision_tree, row)
        output.append((dt_file_order[index].split('_')[0], prediction))
        print('Expected=%s, Got=%s' % (row[-1], prediction))
    return output


def find_accuracy(labels):
    num_correct = 0
    for label in labels:
        if label in all_labeled_data:
            num_correct += 1
    return num_correct / len(all_labeled_data)


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
        print("k for task 2 argument missing")
        exit(0)
    k_from_args = int(args.k)
    build_similarity_matrix()
    find_k_nearest_neighbors(k=k_from_args)

found_labels = []

if c_from_args == '1':
    found_labels = k_nearest_neighbor_classifier()
elif c_from_args == '2':
    found_labels = page_rank_classifier()
elif c_from_args == '3':
    found_labels = decision_tree_classifier('tf')

for found_label in found_labels:
    print(found_label)

accuracy = find_accuracy(found_labels)
print(accuracy)
