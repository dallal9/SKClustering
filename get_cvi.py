import sys
sys.path.append('../')
from metafeatures import Meta

import pandas as pd 
import numpy as np
import csv
from ast import literal_eval
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cdist

class CVIPro:
    def __init__(self, file, meta_type):
        self.file = file
        self.meta_type = meta_type

    def extract_metafeature(self):
        mf = Meta(self.file)
        return mf.extract_metafeatures(self.file, self.meta_type)
        
        
    def nn_search(self):

        #1 - Get other metafeatures from knowledge-base & their CVI combinations
        df_meta_db = pd.read_csv("metafeatures.csv")
        df_meta_instance = self.extract_metafeature()
        df_meta_db = df_meta_db.append(df_meta_instance)
        df_meta_db = df_meta_db.iloc[40:138, :]
        
        #2 - Get known CVI combinations for datasets
        combinations = []
        with open('multi_cvi22.tsv') as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t', fieldnames=['dataset', 'cvi', 'nmi'])
            for row in reader:
                combinations.append(row)

        df_combinations = pd.DataFrame(combinations)
        df_combinations['nmi'] = df_combinations['nmi'].apply(pd.np.float64)
        df_combinations['rank'] = df_combinations.groupby('dataset')['nmi'].rank(ascending=False, axis=0)
        df_combinations = df_combinations.groupby('dataset').apply(lambda x: x.sort_values(['cvi']))
        
        #3 - Compute Euclidean distance between instance and other metafeatures
        df_meta_db_val = df_meta_db.loc[:, df_meta_db.columns != 'dataset']
        distance_matrix = cdist(df_meta_db_val, df_meta_db_val, metric = 'euclidean')
        
        instance_index = len(df_meta_db) - 1
        distances = np.trim_zeros(distance_matrix[instance_index])
        distances_sm = np.sort(distances)[0:5]

        #4 - Get closest meta-features by 5-NN & merge rankings
        all_rank = []
        all_cvis = []
        for dist in distances_sm:
            index = np.where(distances == dist)
            ds = str(df_meta_db.iloc[index].dataset.values[0]) 
            all_rank.append(df_combinations.loc[df_combinations['dataset'] == ds]['rank'].values)

            if len(all_cvis) == 0:
                all_cvis = df_combinations.loc[df_combinations['dataset'] == ds]['cvi'].values
               
        #5 - Select and return best CVI combination by NMI Scores
        values = []
        for rn in all_rank:
            if rn != []:
                values.append(rn)

        if len(values) > 0:
            fn_rank = np.mean(values, axis=0)       # Merged final rankings
            top_rank_index = np.min(fn_rank)        # Get highest ranked CVI combo
            fn_cvi = all_cvis[np.where(fn_rank == top_rank_index)][0]
            return fn_cvi
        else:
            print("Can\'t recommend CVI. No correlation data for neighbors")
            return None

