import random
import numpy as np
import pandas as pd
import pygmo as pg
from cvi import Validation
from GeneticMethods import kmeans, meanshift, dbscan, \
    AffinPropagation, SpectralCluster, AgglomerativeCluster, Optics, BirchClustering
from sklearn import metrics
from operator import itemgetter
from sklearn.impute import SimpleImputer
from sdbw import sdbw
import time
import warnings


class AutoClus:

    def __init__(self, dfile, cvi1, cvi2, cvi3,
                 y=False, iterations=10, size=50):
        """
        Class initialized with:
        - dfile: input data file name (with no labels)
        - cvi1, cvi2, cvi3: a tuple of first a string representing cvi name
         and integer value 1 if the score is maximization and -1 if it's minimization (the smaller the better)
        - iterations: number of iterations to be performed by the model
        - size: size of the population used
        """
        data = pd.read_csv(dfile, header=None, na_values='?')

        if y:
            self.y = data.iloc[:, -1]
            self.data = pd.DataFrame(data.iloc[:, :-1])
        else:
            self.data = data

        if self.data.isnull().values.any():
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

        # list to store the evaluation score of the best model in each iteration
        self.scores = []

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
        algorithms = [
            'agglomerative_clustering',
            'dbscan',
            'meanshift',
            'affinity_propagation',
            'spectral_clustering',
            'optics',
            'birch',
            'kmeans'
          ]

        # algorithms.reverse()

        nr_algorithms = len(algorithms)

        p = size // nr_algorithms

        eval('population.extend(self.' +
             algorithms[0] +
             '.generate_pop(p + (size - p * nr_algorithms)))')

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

        # calculate the integer value for the offspring out of the total size (20% out of total population)
        offspring20 = size // 5
        # calculate the integer value for the crossover out of the total size (5%)
        nr_crossover = size // 20

        for iteration in range(self.iterations):  # start the optimization
            population = self.population

            vals12 = []  # empty list to store the output for the first two evaluation metrics
            vals3 = []  # empty list to store the output for the third evaluation metric

            failed_indices = []
            for i in range(size):

                try:
                    # process the cluster of each configuration in the population
                    clustering = population[i][1].fit(data)
                except Exception as e:
                    print(population[i][0], e)
                    failed_indices.append(i)
                    continue

                try:
                    # get the clustering labels
                    labels = list(clustering.labels_)
                    # if the output has one cluster or n clusters, ignore it
                    if len(set(labels)) == 1 or len(set(labels)) >= (len(data) - 1):
                        failed_indices.append(i)
                        continue
                except Exception as e:
                    print('labels ->', e)
                    failed_indices.append(i)
                    continue

                # some algorithms return cluster labels
                # where the label numbering starts from -1
                # we increment such labels with one
                if -1 in labels:
                    labels = list(np.array(labels) + 1)

                # evaluate clustering
                validate = Validation(data, labels)
                metric_values = validate.run_list([cvi1[0], cvi2[0], cvi3[0]])

                if "sdbw" in [cvi1[0], cvi2[0], cvi3[0]]:
                    if population[i][0] in ['agglomerative_clustering', 'dbscan']:
                        # AgglomerativeClustering' object has no attribute 'cluster_centers_'
                        continue
                    else:
                        sdbw_c = sdbw(data, clustering.labels_, clustering.cluster_centers_)
                        metric_values["sdbw"] = sdbw_c.sdbw_score()

                # first two eval metrics
                vals12.append([metric_values[cvi1[0]] * cvi1[1], metric_values[cvi2[0]] * cvi2[1]])
                vals3.append(metric_values[cvi3[0]] * cvi3[1])  # third eval metric

                try:
                    population[i][2] = vals3[-1]
                except:
                    population[i].append(vals3[-1])

            # removing the failed individuals from the population
            if failed_indices:
                for failed in sorted(failed_indices, reverse=True):
                    population.pop(failed)

            # pareto front optimization to order the configurations using the two eval metrics
            ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=vals12)
            ndf.reverse()

            # ## get the top 20% from the total population
            top_20 = []
            count = 0
            for l in ndf:
                for ix in l:
                    top_20.append(population[ix])
                    count += 1
                    if count >= offspring20:
                        break
                if count >= offspring20:
                    break

            # sorting the individuals w.r.t the third CVI
            top_20 = sorted(top_20, key=itemgetter(2), reverse=True)

            # for alg in top_20:
            #     print(alg[0], end=' ')
            # print('\n')

            try:
                score = self.get_nmi_score(top_20[0][1])
            except Exception as e:
                print('NMI ->', e)
                score = 0.0

            self.scores.append(score)

            crossover_individuals, remaining_individuals = self.do_cross_over(top_20, nr_crossover)

            new_population = []
            if crossover_individuals:
                # if crossover was successful
                new_population.extend(crossover_individuals)
                # offspring = self.create_copies(remaining_individuals)
                offspring = 3 * remaining_individuals  # creating 3 copies of each individual
            else:
                offspring = 3 * top_20
                # offspring = self.create_copies(top_20)

            new_population.extend(self.mutation(offspring))
            # self.population = new_population
            self.generate_pop(population=new_population)

        return top_20  # return the final top 20 solutions

    def get_nmi_score(self, model):
        nmi = metrics.normalized_mutual_info_score(self.y, model.labels_)
        return nmi

    # def create_copies(self, individuals):
    #     """
    #     Creates copies of the individuals from top 20 set,
    #     which were not used in the cross-over phase.
    #     The resulting list has size equal to the overall size
    #     of the initial population.
    #     :param individuals: list of individuals
    #     :return:
    #     """
    #     nr_individuals = len(individuals)
    #     nr_top = self.size // 5
    #     if nr_individuals == nr_top:
    #         # if crossover was not successful
    #         return 5 * individuals
    #     else:
    #         nr_crossovered = nr_top - nr_individuals
    #         to_be_added = self.size - nr_crossovered
    #         multiplicator, remainder = to_be_added // nr_individuals, to_be_added % nr_individuals
    #         copied_individuals = multiplicator * individuals
    #         if remainder == 0:
    #             return copied_individuals
    #         else:
    #             return copied_individuals + individuals[:remainder]

    def do_cross_over(self, top_20, nr_crossover):
        """
        Randomly chooses nr_crossover individuals from the top 20 set,
        looks for a similar individual (belonging to the same clustering algorithm)
        and performs crossover on them
        :param top_20: (list) top 20 individuals from the population
        :param nr_crossover: (int) the number of individuals to do crossover on
        :return: (list) resulting individuals after the crossover
                 (list) the rest of the individuals that were not crossovered
        """
        # select nr_crossover individuals from the top_20 independently
        selected_id = np.random.choice(range(len(top_20)), size=nr_crossover)
        selected_individuals = [top_20[ind] for ind in selected_id]
        remaining_individuals = [top_20[ind] for ind in range(len(top_20)) if ind not in selected_id]
        new_individuals = []
        # do crossover with other individuals from the same algorithm
        for individual in selected_individuals:
            model_name = individual[0]
            similar_individual = self.choose_pair(remaining_individuals, model_name)
            if similar_individual is not None:
                # if there was an individual from the same clustering algorithm
                new_individuals.extend(self.cross_over(individual, similar_individual))
            else:
                # if the selected individual was not paired with another one
                # adding it to the remaining_individuals list
                remaining_individuals.append(individual)
                continue

        return new_individuals, remaining_individuals

    @ staticmethod
    def choose_pair(remaining_ind, model_name):
        """
        Looking for a similar individual to perform crossover
        :param remaining_ind: (list) the set of candidate individuals
        :param model_name: (str) the name of the clustering algorithm
        :return: (list) the chosen individual or
                None in case the pair was not found
        """
        not_found = True
        i = 0
        while not_found and i < len(remaining_ind):
            candidate = remaining_ind[i]
            candidate_model = candidate[0]
            if candidate_model == model_name:
                # found a pair
                not_found = False
                chosen_pair = candidate
                remaining_ind.pop(i)
            else:
                i += 1
                chosen_pair = None
        return chosen_pair

    def cross_over(self, pop1, pop2):
        """
        function to do cross-over between two populations 
        """
        # if pop1[0] != pop2[0]:
        #     return None
        # else:
        model = eval("self." + pop1[0])
        pop1, pop2 = model.cross_over(pop1, pop2)
        return [pop1, pop2]

    def mutation(self, population):
        """
        function to do mutation for a population
        """
        new_population = []
        for pop in population:
            # print('before', pop, sep='\n')
            model = eval("self." + pop[0])
            n_pop = pop
            n_pop = model.mutate(n_pop)
            # print('after', n_pop, sep='\n')
            new_population.append(n_pop)

        return new_population


cvi_set = {'baker_hubert_gamma': 1,
           'ball_hall': 1,
           'banfeld_raferty': -1,
           'c_index': -1,
           'calinski_harabasz': 1,
           'davies_bouldin': -1,
           'det_ratio': 1,
           'dunns_index': 1,
           'g_plus_index': 1,
           'i_index': -1,
           'ksq_detw_index': 1,
           'log_det_ratio': 1,
           'log_ss_ratio': 1,
           'mc_clain_rao': -1,
           'modified_hubert_t': 1,
           'pbm_index': 1,
           'point_biserial': 1,
           'r_squared': 1,
           'ratkowsky_lance': 1,
           'ray_turi': -1,
           'root_mean_square': 1,
           's_dbw': -1,
           'sdbw': -1,
           'scott_symons': -1,
           'tau_index': 1,
           'trace_wib': 1,
           'wemmert_gancarski': 1,
           'trace_w': 1,
           'silhouette': 1,
           'xie_beni': -1}

# cvi1 = ('xie_beni', -1)
# cvi2 = ('modified_hubert_t', 1)
# cvi3 = ('banfeld_raferty', -1)
# #
#
# t1 = time.time()
# auto = AutoClus(dfile="./Datasets/processed/hepta.csv",
#                 y=True,
#                 cvi1=cvi1,
#                 cvi2=cvi2,
#                 cvi3=cvi3,
#                 size=80,
#                 iterations=10)  # initialize class object
# top_20 = auto.evaluate_pop()  # evaluate population and return top 20% after n iterations
# print(auto.scores)
# # print(auto.population)
# print(top_20)
# print((time.time() - t1) / 60)
