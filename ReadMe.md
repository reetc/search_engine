## Task 1:
python task1.py -k=[#K Nearest Neighbors] -m=[#Dominant gestures] -gesture_set_list=[comma-seperated list of gestures. No file extensions]

## Task 2:
python task2.py -c=[Classifier: 1. KNN 2. PPR 3. Decision Tree] -k=[#K Nearest Neighbors, required for KNN and PPR]

## Task 3:
python task3.py --L=[Layers] --k=[Hashes per Layer] --t=[Number of similar files to output] --file =[Query file name]  
Ex: python task3.py --L=9 --k=4 --t=11 --file=560.csv

## Task 4:
python task4.py -q=[Query gesture] -r=[List of relevant gestures] -i=[List of irrelevant gestures]  
Ex: python task4.py -q=570 -r=570,570,571,574,575 -i=572,573
