
import argparse
import task4
import task5
import task3
from task3 import LSH,HTable
from task1 import predict_custom
import pandas
from pprint import pprint


def import_data():
    df = pandas.read_csv('PCA_50_latent_features.csv')
    print(df)
    file_order  = list(df.iloc[:, 0])
    print(file_order)

    data = df.iloc[:, 1:]

    data = data.to_numpy()
    print(data)
    return data,file_order

parser = argparse.ArgumentParser()
parser.add_argument("-option", "--option", help="1 for task 4 ; 2 for task 5")
parser.add_argument("-t", "--t", help="number of results needed")
# parser.add_argument("-t", "--t", help="# of Similar files required")
parser.add_argument("-file", "--file", help="Query file")


args = parser.parse_args()

if args.option is None:
    print("option argument missing")
    exit(0)
option = int(args.option)

if args.t is None:
    print("t argument missing")
    exit(0)
t = int(args.t)

if args.file is None:
    print("file argument missing")
    exit(0)
file_key = args.file.split(".")[0]


query = file_key
relevant_gestures = []
nonrelevant_gestures = []
data,file_order=task5.import_data()
query_vec = data[file_order.index(query+"_vector_tfidf.txt")]

# print(file_order,query_vec)

#task 4
if option == 1:
    while(1):

        new_similarity_gesture_pairs=task4.get_modified_results_after_probabilistic_feedback(query, relevant_gestures, nonrelevant_gestures, untagged_results=[])
        print("Outputs:")
        pprint(new_similarity_gesture_pairs)

        input_str = input("Enter q to quit: ")
        if input_str == "q" or input_str == "Q":
            exit(0)
        # while(1):
        input_str = input("Enter comma seperated relevant file keys: ")
        relevant_gestures = input_str.split(",")
        print("Relevant:", relevant_gestures)

        input_str = input("Enter comma seperated nonn-relevant file keys: ")
        nonrelevant_gestures = input_str.split(",")
        print("Non-Relevant:", nonrelevant_gestures)

        #query_vec = new_query
        #print(query_vec)
        # break

#task 5
elif option == 2:
    while(1):

        new_query=task5.modify_query_vec(relevant_gestures,nonrelevant_gestures,query_vec)
        print(new_query)
        res = task3.predict_custom(new_query,t)
        # res = predict_custom(new_query,t)
        for el in res:
            print("Gesture:",el[1],"Score:",el[0])

        input_str = input("Enter q to quit: ")
        if input_str == "q" or input_str == "Q":
            exit(0)
        # while(1):
        input_str = input("Enter comma seperated relevant file keys: ")
        relevant_gestures = input_str.split(",")
        print("Relevant:",relevant_gestures)

        input_str = input("Enter comma seperated nonn-relevant file keys: ")
        nonrelevant_gestures = input_str.split(",")
        print("Non-Relevant:",nonrelevant_gestures)

        query_vec = new_query
        print(query_vec)
        # break