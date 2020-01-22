from EvalClus import evaluate
import pandas as pd
import numpy as np
import csv

'''
This file runs different configurations of kmeans 
Each parameter different values are defined as paramerter name and “_r” at the end 
All results are written to csv file defined in the code , csv_file = "./output/kmeans_eval_out.csv"
'''

estimator = "birch"
config = {"n_clusters": 3, "threshold": 10, "branching_factor": 50}
csv_file = "./output/birch_out.csv"


''' nmi is a flag, when it is set to true the model will only evaluate configurations based on ground truth data
'''
nmi = True

if nmi:
      csv_file = "./output/BestModel/birch_bestmodel.csv"
else:
      csv_file = "./output/birch_out.csv"


count_all=1

n_clusters_r=list(range(2, 40))
count_all*=len(n_clusters_r)

threshold_r= list(np.arange(0.1, 1, 0.1))
count_all*=len(threshold_r)

branching_factor_r = [10, 50, 75, 100, 200]
count_all*=len(branching_factor_r)

count=0

for n_clusters in n_clusters_r:
    for threshold in threshold_r:
            for branching_factor in branching_factor_r:
                config = {"n_clusters": n_clusters, "threshold": threshold, "branching_factor": branching_factor}
                s = evaluate(estimator, config)
                flag = s.run_all(verbose=True,nmi=nmi)
                out = s.res
                d = {}
                # for key in out:

                for dataset in out.keys():
                    d0 = {"dataset": dataset}
                    d1 = out[dataset]
                    d0.update(d1)

                    d0.update(config)
                    if nmi:
                        dcols=dcols=["dataset" , "n_clusters" , "threshold" , "branching_factor" , 'nmi']
                    else:
                        dcols=["dataset" , "n_clusters" , "threshold" , "branching_factor" , 'Baker_Hubert_Gamma', 'Ball_Hall', 'Banfeld_Raferty', 'Davies_Bouldin', 'Dunns_index', 'McClain_Rao', 'PBM_index', 'Ratkowsky_Lance', 'Ray_Turi', 'Scott_Symons', 'Wemmert_Gancarski', 'Xie_Beni', 'c_index', 'det_ratio', 'g_plus_index', 'i_index', 'ksq_detw_index', 'log_det_ratio', 'log_ss_ratio', 'modified_hubert_t', 'point_biserial', 'r_squared', 'root_mean_square',  's_dbw', 'silhouette', 'tau_index', 'trace_w', 'trace_wib', 'IIndex', 'SDBW', 'ari', 'ami', 'nmi','v_measure','silhouette_score','calinski_harabasz_score']
                    with open(csv_file, 'a', newline='') as csvfile:
                            writer = csv.DictWriter(
                                csvfile, delimiter='\t', fieldnames=dcols)
                            dwrite={}
                            for key in dcols:
                                dwrite[key]=d0[key]
                            
                            writer.writerow(dwrite)
                            csvfile.flush()
                count+=1
                print("run "+str(count)+" configs out of "+str(count_all))
