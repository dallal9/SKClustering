from EvalClus import evaluate
import pandas as pd
import csv

'''
This file runs different configurations of AgglomerativeClustering 
Each parameter different values are defined as paramerter name and “_r” at the end 
All results are written to csv file defined in the code , csv_file = "./output/AgglomerativeClustering_eval_out.csv"
'''

estimator = "AgglomerativeClustering"
csv_file = "./output/AgglomerativeClustering_out.csv"

count_all=1

n_clusters_r=list(range(2, 40))

count_all*=len(n_clusters_r)


linkage_r= ["ward", "complete"]#[2, 10, 50]
count_all*=len(linkage_r)

count=0
for linkage in linkage_r:
      for n_clusters in n_clusters_r:
            config = {"n_clusters": n_clusters, "linkage": linkage}
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

                  dcols=["dataset" , "n_clusters" , "linkage" ,'Baker_Hubert_Gamma', 'Ball_Hall', 'Banfeld_Raferty', 'Davies_Bouldin', 'Dunns_index', 'McClain_Rao', 'PBM_index', 'Ratkowsky_Lance', 'Ray_Turi', 'Scott_Symons', 'Wemmert_Gancarski', 'Xie_Beni', 'c_index', 'det_ratio', 'g_plus_index', 'i_index', 'ksq_detw_index', 'log_det_ratio', 'log_ss_ratio', 'modified_hubert_t', 'point_biserial', 'r_squared', 'root_mean_square',  's_dbw', 'silhouette', 'tau_index', 'trace_w', 'trace_wib', 'IIndex', 'SDBW', 'ari', 'ami', 'nmi','v_measure','silhouette_score','calinski_harabasz_score']
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
