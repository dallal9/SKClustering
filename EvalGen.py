import time 
from GeneticClus import AutoClus
from sklearn import metrics
import glob
import pandas as pd
import numpy as np 

from sklearn.cluster import KMeans, MeanShift, DBSCAN, \
    AffinityPropagation, SpectralClustering, AgglomerativeClustering, \
    OPTICS, Birch

path="./Datasets/processed/"
allFiles = glob.glob(path + "*.csv")
'''
create file  for cvis
'''
'''
save output
'''
allFiles=["./Datasets/processed/iris.csv"]
for file_name in allFiles:
    data = pd.read_csv(file_name, header=None,na_values='?')
    y = data.iloc[:,-1]              
    data = data.iloc[:, :-1]

    t1=time.time()
    auto = AutoClus(dfile=file_name,
        y=True ,
        cvi1=["calinski_harabasz",1],
        cvi2=["davies_bouldin",-1],
        cvi3=["banfeld_raferty",-1],
        size=50,
        iterations=3) #initialize class object
    top_20 = auto.evaluate_pop() #evaluate population and return top 20% after n iterations 
    # print((time.time()-t1))
    print(auto.scores[-1])
    print(np.mean(auto.scores))
    print(np.var(auto.scores))

    auto2 = AutoClus(dfile=file_name,
        y=True ,
        cvi1=["i_index",1],
        cvi2=["ratkowsky_lance",1],
        cvi3=["banfeld_raferty",-1],
        size=8,
        iterations=20) #initialize class object
    auto2.size=8
    auto2.population=[]
    models = auto2.generate_pop(population=[])
    models.append(["meanshift1",MeanShift()])
    models.append(["KMeans1",KMeans()])
    models.append(["DBSCAN1",DBSCAN()])
    models.append(["AffinityPropagation1",AffinityPropagation()])
    models.append(["SpectralClustering1",SpectralClustering()])
    models.append(["AgglomerativeClustering1",AgglomerativeClustering()])
    models.append(["OPTICS1",OPTICS()])
    models.append(["Birch1",Birch()])

    nmis=[]
    lables=[]
    for model in models:
        try:
            model[1].fit(data)
            nmi=metrics.normalized_mutual_info_score(y ,model[1].labels_)
            nmis.append(nmi)
            lables.append(model[0])

        except:
            pass

    print(max(nmis))
    print(np.mean(nmis))
    print(np.var(nmis))
    print(nmis)
    print(lables)
    input("?")
