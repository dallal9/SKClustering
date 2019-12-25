import random 
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN 
from sklearn import metrics
import numpy as np 
import pandas as pd 
import csv
import pygmo as pg
from scipy.stats import spearmanr

class kmeans:
    def __init__(self):
        self.params=["n_clusters","algorithm","init","n_init"]

    def generate_pop(self,size=1):
        population=[]
        for i in range(size+1):
            n_clusters= random.choice(list(range(2,101)))
            init =  random.choice(['k-means++', 'random'])
            algorithm =  random.choice(['auto', 'full', 'elkan'])
            n_init = random.choice(list(range(10,25)))
            population.append(["kmeans",KMeans(n_clusters=n_clusters, algorithm = algorithm ,init=init, n_init=n_init)])
        return population

    def mutate(self,pop):
        p = self.generate_pop(size=1)[0]
        L = len(self.params)
        pos = random.randint(0,L-1)
        
        if pos<=3:
            pop[1].n_init= p[1].n_init
        if pos<=2:
            pop[1].init= p[1].init
        if pos<=1:
            pop[1].algorithm= p[1].algorithm
        if pos==0:
            pop[1].n_clusters= p[1].n_clusters
   
        return pop

    def cross_over(self,pop1,pop2):
        p = pop2
        L = len(self.params)
        pos = random.randint(0,L-1)
        
        if pos<=3:
            pop[1].n_init= p[1].n_init
        if pos<=2:
            pop[1].init= p[1].init
        if pos<=1:
            pop[1].algorithm= p[1].algorithm
        if pos==0:
            pop[1].n_clusters= p[1].n_clusters
   
        return pop

class meanshift:
    def __init__(self):
        self.params=["cluster_all","bin_seeding","init","n_init"]

    def generate_pop(self,size=1):
        population=[]
        for i in range(size+1):
            cluster_all= random.choice([True,False])
            bin_seeding =  random.choice([True,False])
            bandwidth =  random.choice([None,1,2,3,4,5])
            max_iter = random.choice([200,300,400,500,600,700])
            population.append(["meanshift",MeanShift(cluster_all=cluster_all, bin_seeding = bin_seeding ,bandwidth=bandwidth, max_iter=max_iter)])
        return population

    def mutate(self,pop):
        p = self.generate_pop(size=1)[0]
        L = len(self.params)
        pos = random.randint(0,L-1)
        
        if pos<=3:
            pop[1].max_iter= p[1].max_iter
        if pos<=2:
            pop[1].bandwidth= p[1].bandwidth
        if pos<=1:
            pop[1].bin_seeding= p[1].bin_seeding
        if pos==0:
            pop[1].cluster_all= p[1].cluster_all
   
        return pop

    def cross_over(self,pop,pop2):
        p = pop2
        L = len(self.params)
        pos = random.randint(0,L-1)
        
        if pos<=3:
            pop[1].max_iter= p[1].max_iter
        if pos<=2:
            pop[1].bandwidth= p[1].bandwidth
        if pos<=1:
            pop[1].bin_seeding= p[1].bin_seeding
        if pos==0:
            pop[1].cluster_all= p[1].cluster_all
   
        return pop

class dbscan:
    def __init__(self):
        self.params=["eps","min_samples","metric","algorithm","leaf_size","p"]

    def generate_pop(self,size=1):
        population=[]
        for i in range(size+1):
            eps= random.choice([0.3,0.5,0.8,1,2,3,4,5,6,7,8,9,10])
            min_samples =  random.choice([5,10,15,20,30,50,100,150,200])
            metric =  random.choice( ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'])
            algorithm = random.choice(["auto","ball_tree","kd_tree","brute"])
            leaf_size=random.choice([5,10,15,20,25,30,40,50,100,150,200])
            p = leaf_size=random.choice([1,2,3])
            population.append(["dbscan",DBSCAN(eps=eps, metric = metric, min_samples=min_samples,algorithm=algorithm, leaf_size=leaf_size,p=p )])
        return population

    def mutate(self,pop):
        p = self.generate_pop(size=1)[0]
        L = len(self.params)
        pos = random.randint(0,L-1)
        if pos <=5:
            pop[1].eps= p[1].eps
        if pos <=4:
            pop[1].metric= p[1].metric
        if pos<=3:
            pop[1].min_samples= p[1].min_samples
        if pos<=2:
            pop[1].algorithm= p[1].algorithm
        if pos<=1:
            pop[1].leaf_size= p[1].leaf_size
        if pos==0:
            pop[1].p= p[1].p
   
        return pop

    def mutate(self,pop,pop2):
        p = pop2
        L = len(self.params)
        pos = random.randint(0,L-1)
        if pos <=5:
            pop[1].eps= p[1].eps
        if pos <=4:
            pop[1].metric= p[1].metric
        if pos<=3:
            pop[1].min_samples= p[1].min_samples
        if pos<=2:
            pop[1].algorithm= p[1].algorithm
        if pos<=1:
            pop[1].leaf_size= p[1].leaf_size
        if pos==0:
            pop[1].p= p[1].p
   
        return pop




class autoclus:
        def __init__(self,dfile=""):
            data = pd.read_csv(dfile, header=None,na_values='?')           
            self.data = data

            self.kmeans=kmeans()
            self.meanshift=meanshift()
            self.dbscan=dbscan()

            self.population=[]
        
        def generate_pop(self,size=30):
            p=int(size/3)
            k=size-(2*p)
            population=[]
            population.extend(self.kmeans.generate_pop(k))
            population.extend(self.meanshift.generate_pop(p))
            population.extend(self.dbscan.generate_pop(p))

            self.population = population
            return population
            
        def evaluate_pop(self):
            
            new_population=[]
            vals12=[]
            vals3=[]
            indx=[]
            for i in range(len(self.population)):
                Metrics={}
                try: 
                    clustering = self.population[i][1].fit(self.data) 
                except:
                    continue

                if len(set(list(clustering.labels_)))==1:
                    continue
                try:
                    sample_size=int(len(self.data)*0.1)
                    if sample_size<100:
                        sample_size=len(self.data)
                    Metrics["davies_bouldin_score"]=metrics.davies_bouldin_score(self.data,  clustering.labels_) 
                    Metrics["silhouette_score"] = metrics.silhouette_score(self.data, clustering.labels_, metric='euclidean', sample_size=sample_size,random_state=0)
                    Metrics["calinski_harabasz_score"]= metrics.calinski_harabasz_score(self.data,  clustering.labels_) 
                except:
                    continue
                indx.append(i)
                vals12.append([Metrics["davies_bouldin_score"],Metrics["silhouette_score"]])
                vals3.append(Metrics["calinski_harabasz_score"])

            ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points =vals12)
            ndf.reverse()
            
            print(ndf)
        def cross_over(self,pop1,pop2):
            if pop1[0] != pop2[0]:
                return None
            else:
                model=eval("self."+pop1[0])
                n_pop = model.mutate(pop1,pop2)
                return n_pop

        def mutation(self,population=[]):
            new_population=[]
            for pop in population:
                model=eval("self."+pop[0])
                n_pop = model.mutate(pop)
                new_population.append(n_pop)

            return new_population 



auto = autoclus(dfile="test.csv")
auto.generate_pop(15)
auto.evaluate_pop()
