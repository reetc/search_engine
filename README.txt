This repository contains the code, outputs and the report for the phase 3 of the project.

Directories
-----------

The files are organized in different directories like this:
1) Code: This directory contains all the program files, and the data files in the "3_class_gesture_data" subfolder.
2) Outputs: The output files for the results that we generate are put here.
3) Report: This contains the pdf report, where we describe our solution and the results that we obtain.

Environment
-----------

Python version required: Python 3 (3.7 or greater)

Libraries required: numpy scipy pandas scikit-learn matplotlib seaborn

(This code has been tested with Python 3.8.3. But we do not use any version-specific libraries. So any version of
Python 3 > 3.7 should work fine. The latest version of the libraries is recommended. But even so, a
"requirements.txt" file has been provided in the code folder so that the exact environment can be emulated.)

Execution Instructions
----------------------

1) Place component folders in a directory named "3_class_gesture_data"

2) Put the training labels into a file named "sample_training_labels.csv".
Similarly, put the test labels used for calculating accuraacy into a file named "all_labels.csv".

3) Generate tf, tfidf vectors using phase 2 code and put them in files "dataset_tf.txt" and "dataset_tfidf.txt"
Other generated files to copy from phase 2 outputs are "file_order_tf.txt", "file_order_tfidf.txt", "latent_features.csv" and the similarity matrices.

4) Install requirements from file: pip install -r requirements.txt

5) Run the tasks:

## Task 1:
python task1.py -k=[#K Nearest Neighbors] -m=[#Dominant gestures] -gesture_set_list=[comma-seperated list of gestures. No file extensions]
Ex: python task1.py -k=3 -m=10 -gesture_set_list=187,188,189

## Task 2:
python task2.py -c=[Classifier: 1. KNN 2. PPR 3. Decision Tree] -k=[#K Nearest Neighbors, required for KNN and PPR]
Ex: python task2.py -c=2 -k=20

## Task 3:
python task3.py --L=[Layers] --k=[Hashes per Layer] --t=[Number of similar files to output] --file =[Query file name]
Ex: python task3.py --L=9 --k=4 --t=11 --file=187.csv

## Task 4:
python task4.py -q=[Query gesture] -r=[List of relevant gestures] -i=[List of irrelevant gestures]
Ex: python task4.py -q=187 -r=187-0,187-1 -i=644

## Task 5:
python task4.py -q=[Query gesture] -r=[List of relevant gestures] -i=[List of irrelevant gestures]  -t=[no of results to display] -k=[No of nodes to connect in the graph]
Ex: python task5.py -q=187 -r=187-0,187-1 -i=644 -t=10 -k=15

## Task 6:
python task6.py