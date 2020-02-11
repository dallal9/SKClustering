import time 
from GeneticClus import AutoClus
from sklearn import metrics
import glob
import pandas as pd
import numpy as np 
from operator import itemgetter
import csv
from get_cvi import CVIPro
from sklearn.cluster import KMeans, MeanShift, DBSCAN, \
    AffinityPropagation, SpectralClustering, AgglomerativeClustering, \
    OPTICS, Birch


path="./Datasets/processed/"
#allFiles = glob.glob(path + "*.csv")

allFiles=['glass','iono','DS-850','cure-t1-2000n-2D','threenorm','compound','complex9','cluto-t5-8k','complex8','chainlink','wingnut','pathbased','impossible','xclara','disk-4000n','simplex','dense-disk-3000','fourty','smile2','ds850','cure-t0-2000n-2D','cluto-t7-10k','smile3','disk-5000n','jain','smile1','spiral','2dnormals','sonar','triangle1','disk-4600n','sizes3','sizes2','DS-577','atom','long1','dartboard1','flame','triangle2']
for label  in allFiles:
    file_name=path+label+".csv"
    benz = CVIPro(file_name, "distance")
    cvi1,cvi2,cvi3=    eval(benz.nn_search())
    data = pd.read_csv(file_name, header=None,na_values='?')
    y = data.iloc[:,-1]              
    data = data.iloc[:, :-1]

    t1=time.time()
    auto = AutoClus(dfile=file_name,
        y=True ,
        cvi1=[cvi1],
        cvi2=[cvi2],
        cvi3=[cvi3],
        size=35,
        iterations=10,
        label=label) #initialize class object
    top_20 = auto.evaluate_pop() #evaluate population and return top 20% after n iterations 
    # print((time.time()-t1))

    auto2 = AutoClus(dfile=file_name,
        y=True ,
        cvi1=["i_index",1],
        cvi2=["ratkowsky_lance",1],
        cvi3=["banfeld_raferty",-1],
        size=14,
        iterations=3) #initialize class object
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
    labels=[]
    t=[]
    for model in models:
        try:
            model[1].fit(data)
            nmi=metrics.normalized_mutual_info_score(y ,model[1].labels_)
            nmis.append(nmi)
            labels.append(model[0])
            t.append([nmi,model[0],model[1]])

        except:
            pass


        out = open("output/evaluation/out.csv","a",newline='')
        writer = csv.writer(out,delimiter='\t')
        writer.writerow([label,auto.scores[-1],max(auto.scores),np.mean(auto.scores),np.var(auto.scores),str(top_20[0][1]).replace('\n', ' '),max(nmis),np.mean(nmis),np.var(nmis),str(t[0]).replace('\n', ' ')])
