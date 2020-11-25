## Task 1:
python task1.py -k=[#K Nearest Neighbors] -m=[#Dominant gestures] -gesture_set_list=[comma-seperated list of gestures. No file extensions]  
Ex: `python task1.py -k=3 -m=10 -gesture_set_list=1,3,5`  
Ex: `python task1.py -k=5 -m=10 -gesture_set_list=1,3,5`

## Task 2:
python task2.py -c=[Classifier: 1. KNN 2. PPR 3. Decision Tree] -k=[#K Nearest Neighbors, required for KNN and PPR]  
Ex: `python task2.py -c=1 -k=20`  
Ex: `python task2.py -c=2 -k=20`  
Ex: `python task2.py -c=3`

## Task 3:
python task3.py --L=[Layers] --k=[Hashes per Layer] --t=[Number of similar files to output] --file =[Query file name]  
Ex: `python task3.py --L=9 --k=4 --t=11 --file=560.csv`  
Ex: `python task3.py --L=9 --k=4 --t=11 --file=570.csv`  
Ex: `python task3.py --L=9 --k=4 --t=11 --file=270.csv`  
Ex: `python task3.py --L=9 --k=4 --t=11 --file=270-1.csv`  

## Task 4:
python task4.py -q=[Query gesture] -r=[List of relevant gestures] -i=[List of irrelevant gestures]  
Ex: `python task4.py -q=570 -r=570,570,571,574,575 -i=572,573`
Ex: `python task4.py -q=270 -r=270 -i=272`
Ex: `python task4.py -q=270-1 -r=270-1 -i=269-1,269-2`

## Task 5:
python task4.py -q=[Query gesture] -r=[List of relevant gestures] -i=[List of irrelevant gestures]  -t=[no of results to display] -k=[No of nodes to connect in the graph]    
Ex: `python task5.py -q=570 -r=570,570,571,574,575 -i=572,573 -t=10 -k=15`

## Task 6:
`python task6.py`