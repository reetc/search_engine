
import argparse
import task4
import task5
import task3
from task3 import LSH,HTable
# from task1 import predict_custom
import pandas
from pprint import pprint
import sys

def get_similar_gestures_using_task_3(layers, hashes_per_layer, no_of_similar_files, query_file):
    "Returns the similar files as a list of strings without the '.csv' extension E.g. ['570', '571', '574', '575']"
    return task3.get_similar_gestures(query_file, l=layers, k=hashes_per_layer, t=no_of_similar_files)

def run_task_4(q, r:list, i:list, a:list):
    return task4.get_modified_results_after_probabilistic_feedback(query_gesture_name=q, relevant_results=r,
                                                                   irrelevant_results=i, all_results=a)

if __name__ == "__main__":
    query, layers, hashes_per_layer, no_of_similar_files = None, None, None, None
    relevant_gestures = []
    irrelevant_gestures = []
    similar_gestures = []

    task_to_run = input("Enter the task number to run [4,5]: ").strip()

    while True:
        if task_to_run == '4':
            # When you are running it for the first time
            if query is None:
                query = input("Enter the query gesture (without .csv extension): ").strip()
                search_inputs = input("Enter no. of layers, hashes-per-layer, no. of similar files to find "
                                      "(Default 9,4,11): ").strip()
                if search_inputs != "":
                    layers, hashes_per_layer, no_of_similar_files = tuple([int(x.strip()) for x in
                                                                           search_inputs.split(',')])
                else:
                    layers, hashes_per_layer, no_of_similar_files = 9,4,11
                similar_gestures = get_similar_gestures_using_task_3(layers, hashes_per_layer, no_of_similar_files,
                                                                     query)
                print("Similar gestures for", query + ":")
                print("\t" + str(similar_gestures))

            new_relevant_gestures = input("Mark RELEVANT gestures (comma-separated): ").strip()
            if new_relevant_gestures != '':
                relevant_gestures += [x.strip() for x in new_relevant_gestures.split(",")]
            new_irrelevant_gestures = input("Mark IRRELEVANT gestures (comma-separated): ").strip()
            if new_irrelevant_gestures != '':
                irrelevant_gestures += [x.strip() for x in new_irrelevant_gestures.split(",")]
            print("Relevant gestures now:", relevant_gestures)
            print("Irrelevant gestures now:", irrelevant_gestures)
            print("Similar gestures now:", similar_gestures)

            gesture_similarity_series = run_task_4(query, relevant_gestures, irrelevant_gestures, similar_gestures)
            print("Revised similar gestures for", query + ".csv")
            print("--------------------------------")
            print("Gesture Name\t", "Similarity")
            print("--------------------------------")
            for similarity, gesture_name in gesture_similarity_series:
                print(gesture_name + "\t\t\t\t", similarity)
            print('#########################################################################################')

        elif task_to_run == '5':
            # When you are running it for the first time
            if query is None:
                query = input("Enter the query gesture (without .csv extension): ").strip()
                search_inputs = input("Enter no. of layers, hashes-per-layer, no. of similar files to find "
                                      "(Default 9,4,11): ").strip()
                if search_inputs != "":
                    layers, hashes_per_layer, no_of_similar_files = tuple([int(x.strip()) for x in
                                                                           search_inputs.split(',')])
                else:
                    layers, hashes_per_layer, no_of_similar_files = 9, 4, 11
                similar_gestures = get_similar_gestures_using_task_3(layers, hashes_per_layer, no_of_similar_files,
                                                                     query)
                print("Similar gestures for", query + ":")
                print("\t" + str(similar_gestures))

            new_relevant_gestures = input("Mark RELEVANT gestures (comma-separated): ").strip()
            if new_relevant_gestures != '':
                relevant_gestures += [x.strip() for x in new_relevant_gestures.split(",")]
            new_irrelevant_gestures = input("Mark IRRELEVANT gestures (comma-separated): ").strip()
            if new_irrelevant_gestures != '':
                irrelevant_gestures += [x.strip() for x in new_irrelevant_gestures.split(",")]
            print("Relevant gestures now:", relevant_gestures)
            print("Irrelevant gestures now:", irrelevant_gestures)
            print("Similar gestures now:", similar_gestures)

            gesture_similarity_series = task5.get_modified_results_after_ppr(
                query_gesture_name=query,
                relevant_results=relevant_gestures, irrelevant_results=irrelevant_gestures, untagged_results=[],
                numberOfResults=no_of_similar_files
            )
            print("Revised similar gestures for", query + ".csv")
            print("--------------------------------")
            print("Gesture Name\t", "Similarity")
            print("--------------------------------")
            for gesture_name, similarity in gesture_similarity_series.iteritems():
                print(gesture_name + "\t\t\t\t", similarity)
            print('#########################################################################################')

        else:
            print("Invalid option")
            sys.exit(0)