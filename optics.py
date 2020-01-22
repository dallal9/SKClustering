from EvalClus import evaluate
import pandas as pd
import numpy as np
import csv

'''
This file runs different configurations of kmeans 
Each parameter different values are defined as paramerter name and “_r” at the end 
All results are written to csv file defined in the code , csv_file = "./output/kmeans_eval_out.csv"
'''

estimator = "optics"
config = {"min_samples": 5, "cluster_method": "auto", "p": 1, "n_jobs": 1}
csv_file = "./output/optics_out.csv"

''' nmi is a flag, when it is set to true the model will only evaluate configurations based on ground truth data
'''
nmi = True

if nmi:
      csv_file = "./output/BestModel/optics_bestmodel.csv"
else:
      csv_file = "./output/optics_out.csv"


count_all=1

min_samples_r = list(range(2, 20))
count_all*=len(min_samples_r)

cluster_method_r = ["xi", "dbscan"]
count_all*=len(cluster_method_r)

n_jobs_r = [None]
count_all*=len(n_jobs_r)

p_r = [1, 2]
count_all*=len(p_r)

count=0

for min_samples in min_samples_r:
    for cluster_method in cluster_method_r:
            for p in p_r:
                for n_jobs in n_jobs_r:
                    config = {"min_samples": min_samples, "cluster_method": cluster_method, "p": p, "n_jobs": n_jobs}
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
                            dcols=["dataset" , "min_samples" , "cluster_method" , "p" , "n_jobs", 'nmi']
                        else:
                            dcols=["dataset" , "min_samples" , "cluster_method" , "p" , "n_jobs", 'Baker_Hubert_Gamma', 'Ball_Hall', 'Banfeld_Raferty', 'Davies_Bouldin', 'Dunns_index', 'McClain_Rao', 'PBM_index', 'Ratkowsky_Lance', 'Ray_Turi', 'Scott_Symons', 'Wemmert_Gancarski', 'Xie_Beni', 'c_index', 'det_ratio', 'g_plus_index', 'i_index', 'ksq_detw_index', 'log_det_ratio', 'log_ss_ratio', 'modified_hubert_t', 'point_biserial', 'r_squared', 'root_mean_square',  's_dbw', 'silhouette', 'tau_index', 'trace_w', 'trace_wib', 'IIndex', 'SDBW', 'ari', 'ami', 'nmi','v_measure','silhouette_score','calinski_harabasz_score']
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
