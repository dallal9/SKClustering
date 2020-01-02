import random 
from sklearn.cluster import KMeans, MeanShift, DBSCAN, \
    AffinityPropagation, SpectralClustering, AgglomerativeClustering, \
    OPTICS, Birch
from sklearn import metrics
import numpy as np
import pandas as pd
import csv
import pygmo as pg
from scipy.stats import spearmanr
from cvi import validation
from sdbw import sdbw

class kmeans:
    def __init__(self,len_data):
        self.params=["n_clusters","algorithm","init","n_init"]
        self.len_data=len_data

    def generate_pop(self,size=1):
        population=[]
        for i in range(size+1):
            n_clusters=random.choice([2,60,70])#random.choice(list(range(2,min(int(0.3*self.len_data),101))))
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

    def cross_over(self,pop,pop2):
        L = len(self.params)
        pos = random.randint(0,L-1)

        if pos<=3:
            pop[1].n_init, pop2[1].n_init = pop2[1].n_init, pop[1].n_init
        if pos<=2:
            pop[1].init, pop2[1].init = pop2[1].init, pop[1].init
        if pos<=1:
            pop[1].algorithm, pop2[1].algorithm= pop2[1].algorithm, pop[1].algorithm
        if pos==0:
            pop[1].n_clusters, pop2[1].n_clusters =pop2[1].n_clusters, pop[1].n_clusters

        return pop,pop2

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

        L = len(self.params)
        pos = random.randint(0,L-1)

        if pos<=3:
            pop[1].max_iter, pop2[1].max_iter = pop2[1].max_iter, pop[1].max_iter
        if pos<=2:
            pop[1].bandwidth, pop2 [1].bandwidth=  pop2[1].bandwidth, pop [1].bandwidth
        if pos<=1:
            pop[1].bin_seeding, pop2[1].bin_seeding  =pop2[1].bin_seeding, pop[1].bin_seeding
        if pos==0:
            pop[1].cluster_all, pop2[1].cluster_all =pop2[1].cluster_all, pop[1].cluster_all

        return pop, pop2

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

    def cross_over(self,pop,pop2):
        L = len(self.params)
        pos = random.randint(0,L-1)
        if pos <=5:
            pop[1].eps, pop2[1].eps = pop2[1].eps, pop[1].eps
        if pos <=4:
            pop[1].metric, pop2[1].metric = pop2[1].metric, pop[1].metric
        if pos<=3:
            pop[1].min_samples, pop2[1].min_samples =  pop[1].min_samples, pop2[1].min_samples
        if pos<=2:
            pop[1].algorithm, pop2[1].algorithm = pop2[1].algorithm, pop[1].algorithm
        if pos<=1:
            pop[1].leaf_size, pop2[1].leaf_size = pop2[1].leaf_size, pop[1].leaf_size
        if pos==0:
            pop[1].p, pop2[1].p = pop2[1].p, pop[1].p 

        return pop, pop2


class AffinPropagation:
    """
    Affinity Propagation
    """
    def __init__(self):
        self.params = ["damping", "max_iter", "affinity"]
        self.L = len(self.params)  # number of parameters

    @staticmethod
    def generate_pop(size=1):
        population = []
        for i in range(size+1):
            damping = random.uniform(0.5, 1)
            max_iter = random.randint(100, 300)
            affinity = random.choice(['euclidean', 'precomputed'])
            population.append(["affinity_propagation",
                               AffinityPropagation(damping=damping, max_iter=max_iter, affinity=affinity)])
        return population

    def mutate(self, pop):
        p = self.generate_pop(size=1)[0]
        L = self.L
        pos = random.randint(0, L-1)

        if pos <= 2:
            pop[1].damping = p[1].damping
        if pos <= 1:
            pop[1].max_iter = p[1].max_iter
        if pos == 0:
            pop[1].affinity = p[1].affinity

        return pop

    def cross_over(self, pop, pop2):
        p = pop2
        L = self.L
        pos = random.randint(0, L-1)

        if pos <= 2:
            pop[1].damping, pop2[1].damping=  pop2[1].damping , pop[1].damping
        if pos <= 1:
            pop[1].max_iter , p[1].max_iter  = pop2[1].max_iter, pop[1].max_iter
        if pos == 0:
            pop[1].affinity , pop2[1].affinity =  pop2[1].affinity , pop[1].affinity

        return pop, pop2


class SpectralCluster:
    """
    Spectral Clustering
    """
    def __init__(self):
        self.params = ["n_clusters", "eigen_solver",
                       "n_init", "gamma", "affinity"]
        self.L = len(self.params)  # number of parameters

    @staticmethod
    def generate_pop(size=1):
        population = []
        for i in range(size+1):
            n_clusters = random.randint(2, 100)
            eigen_solver = random.choice([None, 'arpack', 'lobpcg', 'amg'])
            n_init = random.randint(1, 20)
            gamma = random.uniform(0.5, 3)
            affinity = random.choice(['nearest_neighbors', 'rbf'])
            population.append(["spectral_clustering",
                               SpectralClustering(n_clusters=n_clusters,
                                                  eigen_solver=eigen_solver,
                                                  n_init=n_init,
                                                  gamma=gamma,
                                                  affinity=affinity)])
        return population

    def mutate(self, pop):
        p = self.generate_pop(size=1)[0]
        L = self.L
        pos = random.randint(0, L-1)

        if pos <= 4:
            pop[1].n_clusters = p[1].n_clusters
        if pos <= 3:
            pop[1].eigen_solver = p[1].eigen_solver
        if pos <= 2:
            pop[1].n_init = p[1].n_init
        if pos <= 1:
            pop[1].gamma = p[1].gamma
        if pos == 0:
            pop[1].affinity = p[1].affinity

        return pop

    def cross_over(self, pop, pop2):

        L = self.L
        pos = random.randint(0, L-1)

        if pos <= 4:
            pop[1].n_clusters , pop2[1].n_clusters  = pop2[1].n_clusters , pop[1].n_clusters
        if pos <= 3:
            pop[1].eigen_solver , pop2[1].eigen_solver = pop2[1].eigen_solver , pop2[1].eigen_solver
        if pos <= 2:
            pop[1].n_init , pop2[1].n_init =  pop2[1].n_init , pop[1].n_init
        if pos <= 1:
            pop[1].gamma , pop2[1].gamma = pop2[1].gamma , pop[1].gamma
        if pos == 0:
            pop[1].affinity , pop2[1].affinity =  pop2[1].affinity , pop[1].affinity

        return pop, pop2


class AgglomerativeCluster:
    """
    Agglomerative Clustering
    """
    def __init__(self):
        self.params = ["n_clusters", "linkage", "affinity"]
        self.L = len(self.params)  # number of parameters

    @staticmethod
    def generate_pop(size=1):
        population = []
        for i in range(size+1):
            n_clusters = random.randint(2, 100)
            linkage = random.choice(['ward', 'complete', 'average', 'single'])
            affinity = random.choice(['euclidean', 'l1', 'l2',
                                      'manhattan', 'cosine'])
            population.append(["agglomerative_clustering",
                               AgglomerativeClustering(n_clusters=n_clusters,
                                                       linkage=linkage,
                                                       affinity=affinity)])
        return population

    def mutate(self, pop):
        p = self.generate_pop(size=1)[0]
        L = self.L
        pos = random.randint(0, L-1)

        if pos <= 2:
            pop[1].n_clusters = p[1].n_clusters
        if pos <= 1:
            pop[1].linkage = p[1].linkage
        if pos == 0:
            pop[1].affinity = p[1].affinity

        return pop

    def cross_over(self, pop, pop2):
        L = self.L
        pos = random.randint(0, L-1)

        if pos <= 2:
            pop[1].n_clusters , pop2[1].n_clusters =  pop2[1].n_clusters , pop[1].n_clusters
        if pos <= 1:
            pop[1].linkage , pop2[1].linkage =  pop2[1].linkage , pop[1].linkage
        if pos == 0:
            pop[1].affinity , pop2[1].affinity = pop2[1].affinity , pop[1].affinity

        return pop, pop2


class Optics:
    """
    OPTICS Clustering
    """
    def __init__(self):
        self.params = ["min_samples", "max_eps", "metric",
                       "cluster_method", "algorithm"]
        self.L = len(self.params)  # number of parameters

    @staticmethod
    def generate_pop(size=1):
        population = []
        for i in range(size+1):
            min_samples = random.uniform(0, 1)
            max_eps = random.choice([np.inf, random.uniform(1, 100)])
            metric = random.choice(['cityblock', 'cosine', 'euclidean',
                                    'l1', 'l2', 'manhattan', 'braycurtis', 
                                    'canberra', 'chebyshev', 'correlation',
                                    'dice', 'hamming', 'jaccard', 'kulsinski',
                                    'mahalanobis', 'minkowski', 'rogerstanimoto',
                                    'russellrao', 'seuclidean', 'sokalmichener',
                                    'sokalsneath', 'sqeuclidean', 'yule'])
            cluster_method = random.choice(['xi', 'dbscan'])
            algorithm = random.choice(['auto', 'ball_tree', 'kd_tree', 'brute'])

            population.append(["optics",
                               OPTICS(min_samples=min_samples, max_eps=max_eps, metric=metric,
                                      cluster_method=cluster_method, algorithm=algorithm)])
        return population

    def mutate(self, pop):
        p = self.generate_pop(size=1)[0]
        L = self.L
        pos = random.randint(0, L-1)

        if pos <= 4:
            pop[1].min_samples = p[1].min_samples
        if pos <= 3:
            pop[1].max_eps = p[1].max_eps
        if pos <= 2:
            pop[1].metric = p[1].metric
        if pos <= 1:
            pop[1].cluster_method = p[1].cluster_method
        if pos == 0:
            pop[1].algorithm = p[1].algorithm

        return pop

    def cross_over(self, pop, pop2):

        L = self.L
        pos = random.randint(0, L-1)

        if pos <= 4:
            pop[1].min_samples , pop2[1].min_samples = pop2[1].min_samples , pop[1].min_samples
        if pos <= 3:
            pop[1].max_eps , pop2[1].max_eps = pop2[1].max_eps , pop[1].max_eps 
        if pos <= 2:
            pop[1].metric , pop2[1].metric = pop2[1].metric , pop[1].metric
        if pos <= 1:
            pop[1].cluster_method , pop2[1].cluster_method = pop2[1].cluster_method , pop[1].cluster_method
        if pos == 0:
            pop[1].algorithm , pop[1].algorithm = pop2[1].algorithm , pop[1].algorithm

        return pop, pop2


class BirchClustering:
    """
    Birch Clustering
    """
    def __init__(self):
        self.params = ["threshold", "branching_factor",
                       "compute_labels", "copy"]
        self.L = len(self.params)  # number of parameters

    @staticmethod
    def generate_pop(size=1):
        population = []
        for i in range(size+1):
            threshold = random.uniform(0.2, 2)
            branching_factor = random.randint(1, 100)
            compute_labels = random.choice([True, False])
            copy = random.choice([True, False])

            population.append(["birch",
                               Birch(threshold=threshold, branching_factor=branching_factor,
                                     compute_labels=compute_labels, copy=copy)])
        return population

    def mutate(self, pop):
        p = self.generate_pop(size=1)[0]
        L = self.L
        pos = random.randint(0, L-1)

        if pos <= 3:
            pop[1].threshold = p[1].threshold
        if pos <= 2:
            pop[1].branching_factor = p[1].branching_factor
        if pos <= 1:
            pop[1].compute_labels = p[1].compute_labels
        if pos == 0:
            pop[1].copy = p[1].copy

        return pop

    def cross_over(self, pop, pop2):

        L = self.L
        pos = random.randint(0, L-1)

        if pos <= 3:
            pop[1].threshold , pop2[1].threshold =  pop2[1].threshold , pop[1].threshold
        if pos <= 2:
            pop[1].branching_factor , pop2[1].branching_factor = pop2[1].branching_factor , pop[1].branching_factor
        if pos <= 1:
            pop[1].compute_labels , pop2[1].compute_labels=   pop2[1].compute_labels , pop[1].compute_label
        if pos == 0:
            pop[1].copy , pop2[1].copy  = pop2[1].copy , pop[1].copy


        return pop


class AutoClus:

    def __init__(self,dfile="",cvi1=["davies_bouldin_score",-1],cvi2=["davies_bouldin_score",-1],cvi3=["davies_bouldin_score",-1],iterations = 10,size=50):
        data = pd.read_csv(dfile, header=None, na_values='?')
        self.data = data

        self.kmeans = kmeans(len(data))
        self.meanshift = meanshift()
        self.dbscan = dbscan()
        self.affinity_propagation = AffinPropagation()
        self.spectral_clustering = SpectralCluster()
        self.agglomerative_clustering = AgglomerativeCluster()
        self.optics = Optics()
        self.birch = BirchClustering()

        
        self.cvi1=cvi1
        self.cvi2=cvi2
        self.cvi3=cvi3


        self.population = []

        self.iterations= iterations
        self.size=size

    def generate_pop(self, population = [] , size=None):
        """

        :param size: size needs to be divisible by # of clustering algorithms
        :return:
        """
        if not size:
            size=self.size
        nr_algorithms = 8  # number of clustering algorithms
        p = int(size / nr_algorithms)
        
        population.extend(self.kmeans.generate_pop(p+(size-(p*8))))
        population.extend(self.meanshift.generate_pop(p))
        population.extend(self.dbscan.generate_pop(p))
        population.extend(self.affinity_propagation.generate_pop(p))
        population.extend(self.spectral_clustering.generate_pop(p))
        population.extend(self.agglomerative_clustering.generate_pop(p))
        population.extend(self.optics.generate_pop(p))
        population.extend(self.birch.generate_pop(p))

        self.population = population

        
        return population

    def evaluate_pop(self):
        offspring20=int(len(self.population)/5)
        crossover5= int(offspring20/4)
        for iteration in range(self.iterations):
            new_population = []
            vals12 = []
            vals3 = []
            indx = []
            
            for i in range(len(self.population)):

                Metrics = {}
                try:
                    clustering = self.population[i][1].fit(self.data)
                except:
                    continue
                
                try:
                    if len(set(list(clustering.labels_))) == 1 or len(set(list(clustering.labels_)))>=(len(self.data)-1):
                        continue
                except:
                    continue
                try:
                    sample_size = int(len(self.data)*0.1)
                    if sample_size < 100:
                        sample_size = len(self.data)
                    labels = list(clustering.labels_)

                    for u in range(len(labels)):
                        if labels[u]<0:
                            labels[u]=0


                    v= validation(np.asmatrix(self.data).astype(np.float),labels )
                    Metrics=v.run_list([self.cvi1[0],self.cvi2[0],self.cvi3[0]])
                    if "SDBW" in [self.cvi1[0],self.cvi2[0]]:
                        sdbw_c = sdbw(self.data, clustering.labels_, clustering.cluster_centers_)
                        Metrics["SDBW"] = sdbw_c.sdbw_score()

                    indx.append(i)
                    vals12.append([Metrics[self.cvi1[0]]*self.cvi1[1],Metrics[self.cvi2[0]]*self.cvi2[1]])
                    vals3.append(Metrics[self.cvi3[0]]*self.cvi3[1])
                    
                except:
                    continue

            ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points =vals12)
            ndf.reverse() 
            #eval3_ind=sorted(range(len(vals3)), key=lambda k: vals3[k])

                
            top_20=[]
            count=0
            for l in ndf:
                for ix in l:
                        top_20.append(self.population[indx[ix]])
                        count+=1
                        if count >= offspring20:
                            break
                if count >= offspring20:
                    break

            for c in range(0,crossover5-2,2):
                new_population.extend(self.cross_over(top_20[c],top_20[c+1]))

            for m in range(crossover5,offspring20):
                if random.randint(1,3)==1:
                    new_population.extend(self.mutation([top_20[m]]))
                else:
                    new_population.append(top_20[m])
            self.population=[]        
            self.generate_pop(population=new_population,size=(self.size-len(new_population )))
        print(len(self.population))
        print(top_20[:3])

    def cross_over(self, pop1, pop2):
        if pop1[0] != pop2[0]:
            return None
        else:
            model = eval("self."+pop1[0])
            pop1,pop2 = model.cross_over(pop1, pop2)
            return [pop1,pop2]

    def mutation(self, population=[]):
        new_population = []
        for pop in population:
            model = eval("self."+pop[0])
            n_pop=pop[:]
            n_pop = model.mutate(n_pop)
            new_population.append(n_pop)

        return new_population

auto = AutoClus(dfile="test.csv", cvi1=["i_index",1],cvi2=["Ratkowsky_Lance",1],cvi3=["Banfeld_Raferty",1],size=50)
auto.generate_pop()
auto.evaluate_pop()
#p0 = auto.mutation(auto.population)
#p1,p2 = auto.cross_over(auto.population[0],auto.population[1])

