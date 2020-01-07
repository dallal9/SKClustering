import random 
from sklearn import metrics
import numpy as np
import pandas as pd
import csv
import pygmo as pg
from scipy.stats import spearmanr
from cvi import validation
from sdbw import sdbw
from GeneticMethods import kmeans, meanshift, dbscan, AffinPropagation, SpectralCluster, AgglomerativeCluster, Optics, BirchClustering


class AutoClus:

    def __init__(self,dfile="",iterations = 10,size=50,cvi1=["davies_bouldin_score",-1],cvi2=["davies_bouldin_score",-1],cvi3=["davies_bouldin_score",-1]):
        """
        Class initialized with:
        - dfile: input data file name (with no labels)
        - cvi1, cvi2, cvi3: a list of first a string representing cvi name and integer value 1 if the score is maximization and -1 if it's minimzation (the smaller the better) 
        - iterations: number of iterations to be performed by the model
        - size: size of the population used
        """
        data = pd.read_csv(dfile, header=None, na_values='?') 
        self.data = data

        #initialize each cluster method to be used by the genetic optimization code.
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
        generate population 
        - if a "population" list is passed the function will increment to it new population generated making the full size equals "size" or "self.size"
        """
        if not size: #if the size of the population is not 
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
        offspring20=int(len(self.population)/5) #calculate the integer value for the offspring out of the total size (20% out of total population)
        crossover5= int(offspring20/4) # calculate the integer value for the crossover out of the total size (5%) 

        for iteration in range(self.iterations): #start the optimization

            new_population = []

            vals12 = [] #empty list to store the output for the first two evaluation metrics 
            vals3 = [] #empty list to store the output for the third evaluation metric
            indx = [] #empty list to store the index of the successful configuration from the population
            
            for i in range(len(self.population)):

                Metrics = {}
                try:
                    clustering = self.population[i][1].fit(self.data) #process the cluster of each configuration in the population
                except:
                    continue
                
                try:
                    if len(set(list(clustering.labels_))) == 1 or len(set(list(clustering.labels_)))>=(len(self.data)-1): #if the output has one cluster or n clusters, ignore it
                        continue
                except:
                    continue
                try:
                    #get the clustering labels
                    sample_size = int(len(self.data)*0.1)
                    if sample_size < 100:
                        sample_size = len(self.data)
                    labels = list(clustering.labels_)

                    for u in range(len(labels)):
                        if labels[u]<0:
                            labels[u]=0

                    #evaluate clustering 
                    v= validation(np.asmatrix(self.data).astype(np.float),labels )
                    Metrics=v.run_list([self.cvi1[0],self.cvi2[0],self.cvi3[0]])
                    if "SDBW" in [self.cvi1[0],self.cvi2[0]]:
                        sdbw_c = sdbw(self.data, clustering.labels_, clustering.cluster_centers_)
                        Metrics["SDBW"] = sdbw_c.sdbw_score()

                    indx.append(i)
                    vals12.append([Metrics[self.cvi1[0]]*self.cvi1[1],Metrics[self.cvi2[0]]*self.cvi2[1]]) #first two eval metrics
                    vals3.append(Metrics[self.cvi3[0]]*self.cvi3[1]) # third eval metric
                    
                except:
                    continue

            ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points =vals12) #pareto front optimization to order the configurations using the two eval mertics.
            ndf.reverse() 

            #get the top 20% from the total population    
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

            #do cross over        
            for c in range(0,crossover5-2,2):
                new_population.extend(self.cross_over(top_20[c],top_20[c+1]))

            #do mutation
            for m in range(crossover5,offspring20):
                if random.randint(1,3)==1:
                    new_population.extend(self.mutation([top_20[m]]))
                else:
                    new_population.append(top_20[m])

            self.population=[] 

            #update population and start new iteration       
            self.generate_pop(population=new_population,size=(self.size-len(new_population )))

        return top_20 #return the final top 20 solutions 

    def cross_over(self, pop1, pop2): 
        '''
        function to do cross-over between two populations 
        '''
        if pop1[0] != pop2[0]:
            return None
        else:
            model = eval("self."+pop1[0])
            pop1,pop2 = model.cross_over(pop1, pop2)
            return [pop1,pop2]

    def mutation(self, population=[]):
        '''
        function to do mutation for a population
        '''
        new_population = []
        for pop in population:
            model = eval("self."+pop[0])
            n_pop=pop[:]
            n_pop = model.mutate(n_pop)
            new_population.append(n_pop)

        return new_population

auto = AutoClus(dfile="test.csv", cvi1=["i_index",1],cvi2=["Ratkowsky_Lance",1],cvi3=["Banfeld_Raferty",1],size=50) #initialize class object
auto.generate_pop() #generate random population
auto.evaluate_pop() #evaluate population and return top 20% after n iterations 
