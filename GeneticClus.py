import random 
from sklearn import metrics
import numpy as np
import pandas as pd
import csv
import pygmo as pg
from scipy.stats import spearmanr
from cvi import Validation
from sdbw import sdbw
from GeneticMethods import kmeans, meanshift, dbscan, AffinPropagation, SpectralCluster, AgglomerativeCluster, Optics, BirchClustering
from sklearn import metrics
from operator import itemgetter
from sklearn.impute import SimpleImputer
import time
import warnings
class AutoClus:

    def __init__(self, dfile, cvi1, cvi2, cvi3,y=False, iterations=10, size=50):
        """
        Class initialized with:
        - dfile: input data file name (with no labels)
        - cvi1, cvi2, cvi3: a list of first a string representing cvi name
         and integer value 1 if the score is maximization and -1 if it's minimization (the smaller the better)
        - iterations: number of iterations to be performed by the model
        - size: size of the population used
        """
        data = pd.read_csv(dfile, header=None, na_values='?') 


        if y:
            self.y = data.iloc[:,-1]
            self.data = pd.DataFrame(data.iloc[:, :-1])
            
        else:
            self.data = data

        if  self.data.isnull().values.any():
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            imp = imp.fit(self.data)
            self.data = pd.DataFrame(imp.transform(self.data))


        # initialize each cluster method to be used by the genetic optimization code.
        self.kmeans = kmeans(len(data))
        self.meanshift = meanshift()
        self.dbscan = dbscan()
        self.affinity_propagation = AffinPropagation()
        self.spectral_clustering = SpectralCluster()
        self.agglomerative_clustering = AgglomerativeCluster()
        self.optics = Optics()
        self.birch = BirchClustering()
        
        self.cvi1 = cvi1
        self.cvi2 = cvi2
        self.cvi3 = cvi3

        self.iterations = iterations
        self.size = size

        # generate the initial random population
        self.population = self.generate_pop()

        warnings.filterwarnings("ignore")

    def generate_pop(self, population=None):
        """
        generate population 
        - if a "population" list is passed the function will increment
        to it new population generated making the full size equals "size" or "self.size"
        """
        if population is None:
            population = []
            size = self.size
        else:
            size = self.size - len(population)
        
        # clustering algorithms
        algorithms = ['kmeans',
                      'meanshift',
                      'dbscan',
                       'affinity_propagation',
                       'spectral_clustering',  
                       'agglomerative_clustering',
                       'optics',
                       'birch'
                      ]

        nr_algorithms = len(algorithms)

        p = size // nr_algorithms
        
        population.extend(self.kmeans.generate_pop(p + (size - p * nr_algorithms)))

        for algorithm in algorithms[1:]:
            eval('population.extend(self.' +
                 algorithm +
                 '.generate_pop(p))')

        self.population = population
        
        return population

    def evaluate_pop(self):

        cvi1 = self.cvi1
        cvi2 = self.cvi2
        cvi3 = self.cvi3
        data = self.data
        size = self.size

        self.scores=[] #list to store the evaluation score of the best model in each iteration

        # calculate the integer value for the offspring out of the total size (20% out of total population)
        offspring20 = size // 5
        # calculate the integer value for the crossover out of the total size (5%)
        crossover5 = size // 20

        for iteration in range(self.iterations):  # start the optimization
            population = self.population
            new_population = []

            vals12 = []  # empty list to store the output for the first two evaluation metrics
            vals3 = []  # empty list to store the output for the third evaluation metric
            indx = []  # empty list to store the index of the successful configuration from the population
            
            for i in range(size):

                try:
                # process the cluster of each configuration in the population
                    clustering = population[i][1].fit(data)
                except:
                    continue

                try:
                    # get the clustering labels
                    labels = list(clustering.labels_)
                    # if the output has one cluster or n clusters, ignore it
                    if len(set(labels)) == 1 or len(set(labels)) >= (len(data)-1):
                        continue
                except:
                     continue

                # try:
                sample_size = int(len(data)*0.1)  # what is the use of this part???
                if sample_size < 100:             #
                    sample_size = len(data)       #

                # some algorithms return cluster labels
                # where the label numbering starts from -1
                # we increment such labels with one,
                # otherwise (in case of the old solution)
                # we have 0 labels more than needed
                if -1 in labels:
                    labels = list(np.array(labels) + 1)
                # for u in range(len(labels)):
                #     if labels[u] < 0:
                #         labels[u] = 0

                # evaluate clustering
                validate = Validation(np.asmatrix(data).astype(np.float), labels)
                metric_values = validate.run_list([cvi1[0], cvi2[0], cvi3[0]])
                if "SDBW" in [cvi1[0], cvi2[0]]:
                    sdbw_c = sdbw(data, clustering.labels_, clustering.cluster_centers_)
                    metric_values["SDBW"] = sdbw_c.sdbw_score()

                indx.append(i)
                # first two eval metrics
                vals12.append([metric_values[cvi1[0]]*cvi1[1], metric_values[cvi2[0]]*cvi2[1]])
                vals3.append(metric_values[cvi3[0]]*cvi3[1])  # third eval metric
                try:
                    self.population[i][2]=metric_values[cvi3[0]]*cvi3[1]
                except:
                    self.population[i].append(metric_values[cvi3[0]]*cvi3[1])
                indx.append(i)
                # except:
                #     continue
            # pareto front optimization to order the configurations using the two eval metrics
            ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=vals12)
            ndf.reverse() 

            # get the top 20% from the total population
            top_20 = []
            count = 0
            for l in ndf:
                for ix in l:
                    top_20.append(population[indx[ix]])
                    count += 1
                    if count >= offspring20:
                        break
                if count >= offspring20:
                    break
                    
            top_20=sorted(top_20, key=itemgetter(2),reverse=True)

            try:
                score=self.get_nmi_score(top_20[0][1])
                
            except:
                score=0.0

            self.scores.append(score)  

            # do cross over
            for c in range(0, crossover5-2, 2):
                new_population.extend(self.cross_over(top_20[c], top_20[c+1]))

            # do mutation
            for m in range(crossover5, offspring20):
                if random.randint(1, 3) == 1:
                    new_population.extend(self.mutation([top_20[m]]))
                else:
                    new_population.append(top_20[m])

            self.population = []

            # update population and start new iteration
            self.generate_pop(population=new_population)
            
        return top_20  # return the final top 20 solutions
    def get_nmi_score(self,model):
        nmi=metrics.normalized_mutual_info_score(self.y ,model.labels_)
        return nmi

    def cross_over(self, pop1, pop2):
        """
        function to do cross-over between two populations 
        """
        if pop1[0] != pop2[0]:
            return None
        else:
            model = eval("self."+pop1[0])
            pop1, pop2 = model.cross_over(pop1, pop2)
            return [pop1, pop2]

    def mutation(self, population):
        """
        function to do mutation for a population
        """
        new_population = []
        for pop in population:
            model = eval("self."+pop[0])
            n_pop = pop[:]
            n_pop = model.mutate(n_pop)
            new_population.append(n_pop)

        return new_population




# t1=time.time()
# auto = AutoClus(dfile="./Datasets/processed/jain.csv",y=True ,cvi1=["i_index",1],cvi2=["ratkowsky_lance",1],cvi3=["banfeld_raferty",-1],size=50,iterations=10) #initialize class object
# top_20 = auto.evaluate_pop() #evaluate population and return top 20% after n iterations 
# print(auto.scores)
# print((time.time()-t1))