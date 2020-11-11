import pandas
import collections
import math
import numpy as np
import argparse


## Import data
def import_data():
    df = pandas.read_csv('latent_features.csv')
    print(df)
    file_order  = list(df.iloc[:, 0])
    print(file_order)

    data = df.iloc[:, 1:]

    data = data.to_numpy()
    print(data)
    return data,file_order






## Hash Table Class

class HTable:
    def __init__(self, k, d):
        self.k = k
        self.d = d
        self.hash_table = dict()
        self.proj = np.random.randn(self.k, d)

    def generate_hash(self, input_vec):
        bools = (np.dot(input_vec, self.proj.T) > 0).astype('int')
        return ''.join(bools.astype('str'))

    def __setitem__(self, input_vec, label):
        hash_val = self.generate_hash(input_vec)
        self.hash_table[hash_val] = self.hash_table\
            .get(hash_val, list()) + [str(label)]

    def __getitem__(self, input_vec):
        hash_val = self.generate_hash(input_vec)
        print(hash_val)
        return self.hash_table.get(hash_val, [])




## LSH Model Class
class LSH:
    def __init__(self, L, k, inp_dimensions):
        self.L = L
        self.k = k
        self.inp_dimensions = inp_dimensions
        self.tables = list()
        for i in range(self.L):
            self.tables.append(HTable(self.k, self.inp_dimensions))

    def __setitem__(self, inp_vec, label):
        for table in self.tables:
            table[inp_vec] = str(label)

    def __getitem__(self, inp_vec):
        self.buckets_searched = 0
        results = list()
        for table in self.tables:
            if len(table[inp_vec]) > 0:
              self.buckets_searched+=1
            results.extend(table[inp_vec])
        return list(set(results))
    def buckets(self):
        return self.buckets_searched







## Generate buckets

def train(l=8,k=4):

    lsh = LSH(L=l,k=k,inp_dimensions=5)
    data,file_order = import_data()
    for i,vec in enumerate(data):
        label = file_order[i].split("_")[0]
        print(label)
        lsh.__setitem__(vec,label)
        # print(i,vec)
    return lsh,file_order


## Train, Evaluate and Store LSH model
def evaluate(l=8,k=4):
    dic = {}
    dic[0] = (0,31)
    dic[1] = (249,279)
    dic[2] = (559,589)
    ## Finding similar
    lsh,file_order = train(l,k)
    data,file_order = import_data()
    sm = 0
    ln = 0
    prec = 0
    map = collections.defaultdict(list)
    for i,vec in enumerate(data):
      res = lsh.__getitem__(vec)
      print(res)
      pnum = int(file_order[i].split("_")[0])
      print(res)
      ans = []
      similar_files = []
      for j in res:
        cnum = int(j)
        flg = 0
        for buc in dic:
          if dic[buc][0]<=cnum<= dic[buc][1] and dic[buc][0]<=pnum<= dic[buc][1]:
            ans.append(1)
            flg = 1
            break
        if flg == 0:
          ans.append(0)

        similar_files.append(j)
      map[file_order[i].split("_")[0]] = similar_files
      sm+=sum(ans)/len(ans)
      prec+=sum(ans)/30
      ln+=len(ans)
      print(file_order[i],ans,sum(ans)/len(ans))
      print("Number of Files Searched",len(similar_files))
      print("Number of Buckets Searched",)
      print(similar_files)


    ## Created matrix with key [file] and value : vector of similar files
    print(map)
    print("Accuracy: ",sm/93)
    print("Precision: ",prec/93)
    print("average length",ln/93)


    import pickle
    with open('lsh_model', 'wb') as model:
        pickle.dump(lsh, model)










def predict(file_num=278,t=15):
    # file_num = 23


    import pickle
    with open('lsh_model', 'rb') as model:
        lsh = pickle.load(model)
    data,file_order = import_data()
    query_num = str(file_num)
    query_file = query_num+"_vector_tfidf.txt"
    # df = pandas.read_csv('latent_features.csv')
    # print(df)
    # file_order  = list(df.iloc[:, 0])
    # print(file_order)
    # print(file_order.index(query_file))
    query_vec = data[file_order.index(query_file)]
    candidates=lsh.__getitem__(query_vec)
    buckets_searched = lsh.buckets()
    # print(query_vec)
    # candidates = map[query_num]
    similarity_mat = collections.defaultdict(list)


    # parent_vector =
    for candidate in candidates:
      candidate_file = str(candidate)+"_vector_tfidf.txt"
      candidate_vec = data[file_order.index(candidate_file)]
      # print(candidate_vec)

      dot = np.dot(query_vec, candidate_vec)
      # norma = np.linalg.norm(query_vec)
      # normb = np.linalg.norm(candidate_vec)
      # cos = dot / (norma * normb)
      distance = math.sqrt(sum([(a - b) ** 2 for a,b  in zip(query_vec, candidate_vec)]))
      # result = 1 - spatial.distance.cosine(query_vec, candidate_vec)
      similarity_mat[query_num].append((distance,candidate))

    print(sorted(similarity_mat[query_num])[0:t])
    print("Total File Considered",len(candidates))
    print("Total Buckets Searched",buckets_searched)






parser = argparse.ArgumentParser()
parser.add_argument("-L", "--L", help="No. of Layers")
parser.add_argument("-k", "--k", help="No. of hashes per layer")
parser.add_argument("-t", "--t", help="# of Similar files required")
parser.add_argument("-file", "--file", help="Query file")


args = parser.parse_args()

if args.k is None:
    print("k for task 3 argument missing")
    exit(0)
k_from_args = int(args.k)

if args.L is None:
    print("L for task 3 argument missing")
    exit(0)
l_from_args = int(args.L)

evaluate(l_from_args,k_from_args)


if args.t is None:
    print("t for task 3 argument missing")
    exit(0)
t_from_args = int(args.t)


if args.file is None:
    print("file for task 3 argument missing")
    exit(0)
file_from_args = args.file.split(".")[0]



# evaluate()
predict(file_from_args,t_from_args)
