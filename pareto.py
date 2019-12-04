import pygmo as pg
import pandas as pd
import numpy as np
import math
import itertools

class PriorityQueue(object): 
    def __init__(self): 
        self.queue = [] 
  
    def __str__(self): 
        return ' '.join([str(i) for i in self.queue]) 
  
    # for checking if the queue is empty 
    def isEmpty(self): 
        return len(self.queue) == [] 
  
    # for inserting an element in the queue 
    def insert(self, data): 
        self.queue.append(data) 
  
    # for popping an element based on Priority 
    def delete(self): 
        try: 
            max = 0
            for i in range(len(self.queue)): 
                if self.queue[i][1] > self.queue[max][1]: 
                    max = i 
            item = self.queue[max] 
            del self.queue[max] 
            return item 
        except IndexError: 
            print() 
            exit() 

def merge (l1,l2):
    if len(l1)!=len(l2):
        return False
    new_l=[]
    for i in range(len(l1)):
        new_l.append([l1[i],l2[i]])
    return new_l

def create_configs(keys):
    configs=[]
    l1 = list(itertools.permutations( keys,2))
    for each in l1:
        for key in keys:
            if key not in each:
                if [each[0],key,each[1]] not in configs:
                    configs.append([key,each[0],each[1]])
    return configs


ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points = [[0,1],[-1,3],[2.3,-0.2],[1.1,-0.12],[1.1, 2.12],[-70,-100]])
ndf = pg.non_dominated_front_2d(points = [[0,1],[-1,3],[2.3,-0.2],[1.1,-0.12],[1.1, 2.12],[-70,-100]])

print(ndf)

df = pd.read_csv('birch1.tsv', sep = "\t")
print(df.columns)

#eval_labels={"silhouette_score":1,"calinski_harabasz_score":1,"davies_bouldin_score":-1,"SSE":-1,"nSSE":-1}
eval_labels={"silhouette_score":1,"calinski_harabasz_score":1,"davies_bouldin_score":-1}
configs = create_configs(list(eval_labels.keys()))
print(len(configs))
print(len(list(itertools.permutations( list(eval_labels.keys()),3))))
eval_max={}
for key in eval_labels:
    eval_max[key]={"score":-1*math.inf,"ARI":0}
scores={}
for key in eval_labels:
    scores[key]=[]

max_sil={"score":-100,"ARI":0}
max_ch={"score":-100,"ARI":0}
max_db={"score":-100,"ARI":0}

max_ARI=-100

sil_db=[]
ARIs=[]

sils=[]
dbs=[]

chs=[]
for  k, row in df[df["dataset"]=="a1"].iterrows():
    for key in eval_labels:
        score=row[key]*eval_labels[key]
        scores[key].append(score)
    ARI=row["ARI"]
    ARIs.append(ARI)

max_ari=-1
max_label=""
for label in eval_labels.keys(): 
   ari_score= ARIs[scores[label].index(max(scores[label]))]
   if ari_score>max_ari:
       max_ari=ari_score
       max_label=label

print(label,max_ari,max(ARIs))

max_ari=-1
max_label=""
queue = PriorityQueue()
for config in configs: 
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points =merge(scores[config[0]],scores[config[1]]))
    if len (ndf[-1])==1:
        i = ndf[-1][0]
        ari_score=ARIs[i]
        label=str(config)
        
    else:
        max_index=-1
        max_score=-1
        for index in ndf[-1]:
            if scores[config[2]][index]>max_score:
                max_score=scores[config[2]][index]
                max_index=index
        ari_score=ARIs[index]
        label=str(config)
    queue.insert([label,ari_score])

item  =  queue.delete()
print(item)

