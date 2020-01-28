import time 
from GeneticClus import AutoClus
from sklearn import metrics
import glob
import pandas as pd

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

for file_name in allFiles:
    data = pd.read_csv(file_name, header=None,na_values='?')
    y = data.iloc[:,-1]              
    data = data.iloc[:, :-1]

    t1=time.time()
    auto = AutoClus(dfile=file_name,
        y=True ,
        cvi1=["i_index",1],
        cvi2=["ratkowsky_lance",1],
        cvi3=["banfeld_raferty",-1],
        size=50,
        iterations=10) #initialize class object
    top_20 = auto.evaluate_pop() #evaluate population and return top 20% after n iterations 
    # print((time.time()-t1))
    print(auto.scores)

    auto.size=8
    auto.population=[]
    models = auto.generate_pop(population=[])
    models.append(["meanshift",MeanShift()])
    models.append(["KMeans",KMeans()])
    models.append(["DBSCAN",DBSCAN()])
    models.append(["AffinityPropagation",AffinityPropagation()])
    models.append(["SpectralClustering",SpectralClustering()])
    models.append(["AgglomerativeClustering",AgglomerativeClustering()])
    models.append(["OPTICS",OPTICS()])
    models.append(["Birch",Birch()])

    nmis=[]
    for model in models:
        try:
            model[1].fit(data)
            nmi=metrics.normalized_mutual_info_score(y ,model[1].labels_)
            nmis.append(nmi)

        except:
            pass

    print(max(nmis))
    input("?")
