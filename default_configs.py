from sklearn.cluster import KMeans, MeanShift, DBSCAN, \
    AffinityPropagation, SpectralClustering, AgglomerativeClustering, \
    OPTICS, Birch
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from GeneticClus import AutoClus
import glob
import warnings
import time

# def fit_with_default(dataset_name):
#
#     dataset = pd.read_csv(dataset_name, header=None)
#     if dataset.isnull().values.any():
#         imp = SimpleImputer(missing_values=np.nan, strategy='mean')
#         dataset = pd.DataFrame(imp.fit_transform(dataset))
#
#     y = dataset.iloc[:, -1]
#     data = dataset.iloc[:, :-1]
#     algorithms = [
#         KMeans(),
#         MeanShift(),
#         DBSCAN(),
#         AffinityPropagation(),
#         SpectralClustering(),
#         AgglomerativeClustering(),
#         OPTICS(),
#         Birch()
#     ]
#
#     nmi_per_algorithm = {}
#     nmi_per_algorithm['dataset'] = dataset_name
#     for index, algorithm in enumerate(algorithms):
#         print('Algorithm {}'.format(index + 1), flush=True)
#         clustering = algorithm.fit(data)
#         labels = list(clustering.labels_)
#
#         if -1 in labels:
#             labels = list(np.array(labels) + 1)
#
#         nmi = metrics.normalized_mutual_info_score(y, labels)
#         nmi_per_algorithm[str(algorithm).lower().split(sep='(')[0]] = nmi
#
#     return pd.DataFrame(nmi_per_algorithm, index=[0])

warnings.filterwarnings("ignore")

path = './Datasets/processed/'
files = glob.glob(path+'*.csv')

# for i in map(lambda x: x.split('\\')[-1].split('.csv')[0], files):
#     print(i)

already_processed = ['diamond9',
        'disk-3000n',
        'donut3',
        'donutcurves',
        'dpb',
        'DS-850',
        'ds4c2sc8',
        'elliptical_10_2',
        'elly-2d10c13s',
        'hypercube',
        'long3',
        'longsquare',
        'lsun',
        'shapes',
        'sizes1',
        'sizes2',
        'sizes4',
        'sizes5',
        'spherical_4_3',
        'spherical_5_2',
        'square1',
        'square2',
        'square4',
        'square5',
        'target',
        'tetra',
        'threenorm',
        'triangle2',
        'twenty',
        'twodiamonds',
        'zelnik1',
        'zelnik2',
        'zelnik6']

# out_data = pd.DataFrame()
# for index, dataset_name in enumerate(files):
#     print('Dataset {}\n'.format(index + 1), flush=True)
#     try:
#         scores = fit_with_default(dataset_name)
#     except:
#         continue
#     out_data = pd.concat([out_data, scores], axis=0)
#     out_data.to_csv('default.csv', header=True, index=False)

# https://cran.r-project.org/web/packages/clusterCrit/vignettes/clusterCrit.pdf (page 21)
cvi_set = {'baker_hubert_gamma': 1,
           'ball_hall': -1,  #  max diff
           'banfeld_raferty': -1,
           'c_index': -1,
           'calinski_harabasz': 1,
           'davies_bouldin': -1,
           'det_ratio': 1,  #  min diff
           'dunns_index': 1,
           'g_plus_index': -1, #
           'i_index': -1,
           'ksq_detw_index': 1, #  max diff
           'log_det_ratio': 1, #  min diff
           'log_ss_ratio': 1, #  min diff
           'mc_clain_rao': -1,
           'modified_hubert_t': 1,
           'pbm_index': 1,
           'point_biserial': 1,
           'r_squared': 1,
           'ratkowsky_lance': 1,
           'ray_turi': -1,
           'root_mean_square': 1, #?
           's_dbw': -1,
           'sdbw': -1,
           'scott_symons': -1,
           'tau_index': 1,
           'trace_wib': 1, #  max diff
           'wemmert_gancarski': 1,
           'trace_w': 1, # max diff
           'silhouette': 1,
           'xie_beni': -1}

best_cvi = pd.read_csv('best_cvi.csv')

# problematic_cvi = ['baker_hubert_gamma',
#                    'g_plus_index',
#                    'tau_index',
#                    's_dbw',
#                    'trace_wib',
#                    'wemmert_gancarski',
#                    'sdbw']
#
out_data = pd.DataFrame()
row_info = {}
for index, file in enumerate(files):

    dname = file.split('\\')[-1][:-4]

    print('Dataset {}\n'.format(index + 1), flush=True)

    row_info['dataset'] = dname
    if not (best_cvi.dataset == dname).any():
        print('\n', dname, '\n')
        continue

    cvi_list = str(best_cvi.loc[best_cvi.dataset == dname, 'cvi'].values[0][1:-1].lower()).split(', ')
    row_info['cvi'] = str(cvi_list)
    cvi1 = [cvi_list[0], cvi_set[cvi_list[0]]]
    cvi2 = [cvi_list[1], cvi_set[cvi_list[1]]]
    cvi3 = [cvi_list[2], cvi_set[cvi_list[2]]]

    # if not np.isin([cvi1[0], cvi2[0], cvi3[0]],
    #            problematic_cvi).any():
    #     continue

    print(cvi1, cvi2, cvi3, sep=' ', end='\n\n')

    genetic_search = AutoClus(dfile=file,
                              y=True,
                              cvi1=cvi1,
                              cvi2=cvi2,
                              cvi3=cvi3,
                              size=50,
                              iterations=10)

    try:
        start = time.perf_counter()
        top_20 = genetic_search.evaluate_pop()
        time_taken = (time.perf_counter() - start) / 60    # in minutes
    except Exception as e:
        print(e)
        continue
    row_info['algorithm'] = top_20[0][0]
    row_info['nmi'] = genetic_search.scores[0]
    row_info['time_taken'] = time_taken
    row_info = pd.DataFrame(row_info, index=[0])
    out_data = pd.concat([out_data, row_info], axis=0)
    out_data.to_csv('genetic_search.csv', header=True, index=False)

