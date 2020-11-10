import argparse


def k_nearest_neighbor_classifier():
    pass


def page_rank_classifier():
    pass


def decision_tree_classifier():
    pass


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--c",
                    help="Classifier. Options: 1. K-Nearest Neighbor 2. Personalized-PageRank 3. Decision Tree")

args = parser.parse_args()

if args.c is None:
    print("k for task 1 argument missing")
    exit(0)
else:
    if args.c not in ['1','2','3']:
        print("Unrecognized Classifier")
        exit(0)
c_from_args = args.c

if c_from_args == '1':
    k_nearest_neighbor_classifier()
elif c_from_args == '2':
    page_rank_classifier()
elif c_from_args == '3':
    decision_tree_classifier()