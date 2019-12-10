import pygmo as pg
import pandas as pd
import numpy as np
import math
import itertools
from scipy.stats import spearmanr
import csv

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
            return None 

def get_slice(l,index=2):
    new_slice=[]
    for i in l:
        new_slice.append(i[index])
    return new_slice

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

df = pd.read_csv('combined.tsv', sep = "\t")
datasets = list(set(df["dataset"]))

#eval_labels={"silhouette_score":1,"calinski_harabasz_score":1,"davies_bouldin_score":-1,"SSE":-1,"nSSE":-1}
#eval_labels={"silhouette_score":1,"calinski_harabasz_score":1,"davies_bouldin_score":-1}
#eval_labels={"Baker_Hubert_Gamma":-1,"Banfeld_Raferty":-1,"Davies_Bouldin":-1,"Dunns_index":1,"McClain_Rao":-1,"PBM_index":1,"Ratkowsky_Lance":1,"Ray_Turi":-1,"Scott_Symons":-1,"Wemmert_Gancarski":1,"Xie_Beni":-1,"c_index":-1,"g_plus_index":-1,"i_index":1,"modified_hubert_t":1,"point_biserial":1,"s_dbw":-1,"silhouette":1,"tau_index":1,"IIndex":1,"SDBW":-1,"calinski_harabasz_score":1}
eval_labels={"Banfeld_Raferty":-1,"Davies_Bouldin":-1,"Dunns_index":1,"McClain_Rao":-1,"PBM_index":1,"Ratkowsky_Lance":1,"Ray_Turi":-1,"Scott_Symons":-1,"Xie_Beni":-1,"c_index":-1,"i_index":1,"modified_hubert_t":1,"point_biserial":1,"s_dbw":-1,"silhouette":1,"IIndex":1,"SDBW":-1,"calinski_harabasz_score":1}

configs = create_configs(list(eval_labels.keys()))
print(len(configs))
print(len(list(itertools.permutations( list(eval_labels.keys()),3))))
infos=[]
for dataset_name in datasets:
    
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
    for  k, row in df[df["dataset"]==dataset_name].iterrows():
        for key in eval_labels:
            score=row[key]*eval_labels[key]
            scores[key].append(score)
        ARI=row["nmi"]
        ARIs.append(ARI)

    max_ari=-1
    max_label=""
    for label in eval_labels.keys(): 
        ari_score= ARIs[scores[label].index(max(scores[label]))]
    if ari_score>max_ari:
        max_ari=ari_score
        max_label=label

    #print(label,max_ari,max(ARIs))

    max_ari=-1
    max_label=""
    OUT=[]
    for config in configs: 

        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points =merge(scores[config[0]],scores[config[1]]))
        ndf.reverse()

        output=[]
        for l in ndf:
            queue=PriorityQueue()
            max_index=-1
            max_score=-1
            for index in l:
                ari_score=ARIs[index]
                label=str(config)
                queue.insert([label,ari_score,index])
            item =1
            output0=[]
            while item:
                item=queue.delete()
                if item==None:
                    break
                else:
                    output0.append(item)
            output.extend(output0)
        OUT.append(output)
    ARIs_in=sorted(range(len(ARIs)), key=lambda k: ARIs[k],reverse=True)
    infos.append({"dataset":dataset_name,"scores":dict(scores),"ARI_INFO":list(ARIs_in),"OUT":list(OUT)})
print(len(infos))
for info in infos:
    OUT=info["OUT"]
    ARIs_in=info["ARI_INFO"]
    scores=info["scores"]
    dataset=info["dataset"]
    max3=-1
    label=""
    l=[]
    for each in OUT:
        c = spearmanr(get_slice(each),ARIs_in)
        if  c[0] > max3:
            max3 = c[0]
            label=each[0][0]
        #l.append((label,max3))

    #l.sort(key=lambda tup: tup[1]) 
    #print(l)
    #asda
    dwrite={"dataset":dataset,"label":label,"correlation":max3}
    with open("out_pareto.tsv", 'a', newline='') as csvfile:
        writer = csv.DictWriter(  csvfile, delimiter='\t', fieldnames=["dataset","label","correlation"])                          
        writer.writerow(dwrite)
        csvfile.flush()

    print(max3,label)



    #t1 = sorted(range(len(scores["davies_bouldin_score"])), key=lambda k: scores["davies_bouldin_score"][k],reverse=True)
    max1=-10000
    label1=""


    for key in list(eval_labels.keys()):
        t2 = sorted(range(len(scores[key])), key=lambda k: scores[key][k])
        c = spearmanr(t2,ARIs_in)
        
        if c[0] > max1:
            max1=c[0]
            label1=key


    dwrite={"dataset":dataset,"label":label1,"correlation":max1}
    with open("out_pareto2.tsv", 'a', newline='') as csvfile:
        writer = csv.DictWriter(  csvfile, delimiter='\t', fieldnames=["dataset","label","correlation"])                          
        writer.writerow(dwrite)
        csvfile.flush()



    print(max1,key)

    print()