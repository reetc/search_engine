
import argparse
import task4
import task5
import task3
from task3 import LSH,HTable
# from task1 import predict_custom
import pandas
from pprint import pprint
import sys


# def import_data():
#     df = pandas.read_csv('PCA_50_latent_features.csv')
#     print(df)
#     file_order  = list(df.iloc[:, 0])
#     print(file_order)
#
#     data = df.iloc[:, 1:]
#
#     data = data.to_numpy()
#     print(data)
#     return data,file_order
#
# parser = argparse.ArgumentParser()
# parser.add_argument("-option", "--option", help="1 for task 4 ; 2 for task 5")
# parser.add_argument("-t", "--t", help="number of results needed")
# # parser.add_argument("-t", "--t", help="# of Similar files required")
# parser.add_argument("-file", "--file", help="Query file")
#
#
# args = parser.parse_args()
#
# if args.option is None:
#     print("option argument missing")
#     exit(0)
# option = int(args.option)
#
# if args.t is None:
#     print("t argument missing")
#     exit(0)
# t = int(args.t)
#
# if args.file is None:
#     print("file argument missing")
#     exit(0)
# file_key = args.file.split(".")[0]
#
#
# query = file_key
# relevant_gestures = []
# nonrelevant_gestures = []
# data,file_order=task5.import_data()
# query_vec = data[file_order.index(query+"_vector_tfidf.txt")]
#
# # print(file_order,query_vec)
#
#
# #task 4
# if option == 1:
#     while(1):
#
#         new_similarity_gesture_pairs=task4.get_modified_results_after_probabilistic_feedback(query, relevant_gestures, nonrelevant_gestures, untagged_results=[])
#         print("Outputs:")
#         pprint(new_similarity_gesture_pairs)
#
#         input_str = input("Enter q to quit: ")
#         if input_str == "q" or input_str == "Q":
#             exit(0)
#         # while(1):
#         input_str = input("Enter comma seperated relevant file keys: ")
#         relevant_gestures = input_str.split(",")
#         print("Relevant:", relevant_gestures)
#
#         input_str = input("Enter comma seperated nonn-relevant file keys: ")
#         nonrelevant_gestures = input_str.split(",")
#         print("Non-Relevant:", nonrelevant_gestures)
#
#         #query_vec = new_query
#         #print(query_vec)
#         # break
#
# #task 5
# elif option == 2:
#     while(1):
#
#         new_query=task5.modify_query_vec(relevant_gestures,nonrelevant_gestures,query_vec)
#         print(new_query)
#         res = task3.predict_custom(new_query,t)
#         # res = predict_custom(new_query,t)
#         for el in res:
#             print("Gesture:",el[1],"Score:",el[0])
#
#         input_str = input("Enter q to quit: ")
#         if input_str == "q" or input_str == "Q":
#             exit(0)
#         # while(1):
#         input_str = input("Enter comma seperated relevant file keys: ")
#         relevant_gestures = input_str.split(",")
#         print("Relevant:",relevant_gestures)
#
#         input_str = input("Enter comma seperated nonn-relevant file keys: ")
#         nonrelevant_gestures = input_str.split(",")
#         print("Non-Relevant:",nonrelevant_gestures)
#
#         query_vec = new_query
#         print(query_vec)
#         # break

def get_similar_gestures_using_task_3(layers, hashes_per_layer, no_of_similar_files, query_file):
    "Returns the similar files as a list of strings without the '.csv' extension E.g. ['570', '571', '574', '575']"
    return task3.get_similar_gestures(query_file, l=layers, k=hashes_per_layer, t=no_of_similar_files)

def run_task_4(q, r:list, i:list, a:list):
    return task4.get_modified_results_after_probabilistic_feedback(query_gesture_name=q, relevant_results=r,
                                                                   irrelevant_results=i, all_results=a)

def run_task_5():
    pass

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

            similarity_gesture_pairs = run_task_4(query, relevant_gestures, irrelevant_gestures, similar_gestures)
            print("Revised similar gestures for", query + ".csv")
            print("--------------------------------")
            print("Gesture Name\t", "Similarity")
            print("--------------------------------")
            for similarity, gesture_name in similarity_gesture_pairs:
                print(gesture_name + "\t\t\t\t", similarity)
            print('#########################################################################################')

        elif task_to_run == '5':
            run_task_5()

        else:
            print("Invalid option")
            sys.exit(0)