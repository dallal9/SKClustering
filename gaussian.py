from EvalClus import evaluate
import pandas as pd
import numpy as np
import csv

'''
This file runs different configurations of kmeans 
Each parameter different values are defined as paramerter name and “_r” at the end 
All results are written to csv file defined in the code , csv_file = "./output/kmeans_eval_out.csv"
'''

estimator = "gaussian"
config = {"n_components": 1, "covariance_type": "full", "n_init": 1, "init_params": "kmeans"}
csv_file = "./output/gaussian_out.csv"

count_all=1

init_params_r=['random']
count_all*=len(init_params_r)

n_init_r= [1, 2, 5, 10, 20, 50]
count_all*=len(n_init_r)

covariance_type_r =['full', 'tied', 'diag', 'spherical']
count_all*=len(covariance_type_r)

n_components_r=list(range(1, 20))
count_all*=len(n_components_r)


count=0

for n_components in n_components_r:
    for covariance_type in covariance_type_r:
            for n_init in n_init_r:
                for init_params in init_params_r:
                    config = {"n_components": n_components, "covariance_type": covariance_type, "n_init": n_init, "init_params": init_params}
                    s = evaluate(estimator, config)
                    s.run_all(verbose=True)
                    out = s.res
                    d = {}
                    # for key in out:

                    for dataset in out.keys():
                        d0 = {"dataset": dataset}
                        d1 = out[dataset]
                        d0.update(d1)

                        d0.update(config)

                        dcols=["dataset" , "n_components" , "covariance_type" , "n_init" , "init_params", 'Baker_Hubert_Gamma', 'Ball_Hall', 'Banfeld_Raferty', 'Davies_Bouldin', 'Dunns_index', 'McClain_Rao', 'PBM_index', 'Ratkowsky_Lance', 'Ray_Turi', 'Scott_Symons', 'Wemmert_Gancarski', 'Xie_Beni', 'c_index', 'det_ratio', 'g_plus_index', 'i_index', 'ksq_detw_index', 'log_det_ratio', 'log_ss_ratio', 'modified_hubert_t', 'point_biserial', 'r_squared', 'root_mean_square',  's_dbw', 'silhouette', 'tau_index', 'trace_w', 'trace_wib', 'IIndex', 'SDBW', 'ari', 'ami', 'nmi','v_measure','silhouette_score','calinski_harabasz_score']
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
