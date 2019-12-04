from EvalClus import evaluate
import pandas as pd
import csv

'''
This file runs different configurations of DBSCAN 
Each parameter different values are defined as paramerter name and “_r” at the end 
All results are written to csv file defined in the code , csv_file = "./output/DBSCAN_out.csv"
'''

estimator = "DBSCAN"
csv_file = "./output/DBSCAN_out.csv"

count_all=1

leaf_size_r=[90]
count_all*=len(leaf_size_r)

metric_r=['euclidean',  'manhattan']
count_all*=len(metric_r)

eps_r=[10]
count_all*=len(eps_r)

min_samples_r=[2]
count_all*=len(min_samples_r)
count=0

for leaf_size in leaf_size_r:
      for metric in metric_r:
            for eps in eps_r:
                  for min_samples in min_samples_r:
                        config = {"leaf_size": leaf_size, "metric": metric,
                              "eps": eps, "min_samples": min_samples}
                        s = evaluate(estimator, config)
                        
                        flag = s.run_all(verbose=True)
                        if not flag:
                              continue
                        out = s.res
                        d = {}
                        # for key in out:

                        for dataset in out.keys():
                              d0 = {"dataset": dataset}
                              d1 = out[dataset]
                              d0.update(d1)

                              d0.update(config)
      
                              dcols=["dataset" , "n_clusters" , "init" , "max_iter" , "n_init" , "algorithm" ,'Baker_Hubert_Gamma', 'Ball_Hall', 'Banfeld_Raferty', 'Davies_Bouldin', 'Dunns_index', 'McClain_Rao', 'PBM_index', 'Ratkowsky_Lance', 'Ray_Turi', 'Scott_Symons', 'Wemmert_Gancarski', 'Xie_Beni', 'c_index', 'det_ratio', 'g_plus_index', 'i_index', 'ksq_detw_index', 'log_det_ratio', 'log_ss_ratio', 'modified_hubert_t', 'point_biserial', 'r_squared', 'root_mean_square',  's_dbw', 'silhouette', 'tau_index', 'trace_w', 'trace_wib', 'IIndex', 'SDBW', 'ari', 'ami', 'nmi','v_measure','silhouette_score','calinski_harabasz_score']
                              with open(csv_file, 'a', newline='') as csvfile:
                                    writer = csv.DictWriter(
                                          csvfile, delimiter='\t', fieldnames=dcols)
                                    #writer.writeheader()
                                    # for data in dict_data:
                                    dwrite={}
                                    for key in dcols:
                                          dwrite[key]=d0[key]
                                    
                                    writer.writerow(dwrite)
                                    csvfile.flush()
                        count+=1
                        print("run "+str(count)+" configs out of "+str(count_all))
