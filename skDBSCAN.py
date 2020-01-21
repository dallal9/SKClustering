from EvalClus import evaluate
import pandas as pd
import csv

'''
This file runs different configurations of DBSCAN 
Each parameter different values are defined as paramerter name and “_r” at the end 
All results are written to csv file defined in the code , csv_file = "./output/DBSCAN_out.csv"
'''

estimator = "DBSCAN"


''' nmi is a flag, when it is set to true the model will only evaluate configurations based on ground truth data
'''
nmi = True

if nmi:
      csv_file = "./output/BestModel/DBSCAN_bestmodel.csv"
else:
      csv_file = "./output/DBSCAN_out.csv"
count_all=1

leaf_size_r=[50,90,100]
count_all*=len(leaf_size_r)

metric_r=['euclidean',  'manhattan','cosine']
count_all*=len(metric_r)

eps_r=[0.2,0.3,0.5]
count_all*=len(eps_r)

min_samples_r=[2,3,4]
count_all*=len(min_samples_r)
count=0

for leaf_size in leaf_size_r:
      for metric in metric_r:
            for eps in eps_r:
                  for min_samples in min_samples_r:
                        config = {"leaf_size": leaf_size, "metric": metric,
                              "eps": eps, "min_samples": min_samples}
                        s = evaluate(estimator, config)
                        count+=1
                        print("run "+str(count)+" configs out of "+str(count_all))
                        flag = s.run_all(verbose=True,nmi=nmi)
                        if  flag:
                              out = s.res
                              d = {}
                              # for key in out:

                              for dataset in out.keys():
                                    d0 = {"dataset": dataset}
                                    d1 = out[dataset]
                                    d0.update(d1)

                                    d0.update(config)
                                    if nmi:
                                          dcols = ["dataset" , "leaf_size" , "metric" , "eps" , "min_samples",'nmi']
                                    else:
                                          dcols=["dataset" , "leaf_size" , "metric" , "eps" , "min_samples",'Baker_Hubert_Gamma', 'Ball_Hall', 'Banfeld_Raferty', 'Davies_Bouldin', 'Dunns_index', 'McClain_Rao', 'PBM_index', 'Ratkowsky_Lance', 'Ray_Turi', 'Scott_Symons', 'Wemmert_Gancarski', 'Xie_Beni', 'c_index', 'det_ratio', 'g_plus_index', 'i_index', 'ksq_detw_index', 'log_det_ratio', 'log_ss_ratio', 'modified_hubert_t', 'point_biserial', 'r_squared', 'root_mean_square',  's_dbw', 'silhouette', 'tau_index', 'trace_w', 'trace_wib', 'IIndex', 'SDBW', 'ari', 'ami', 'nmi','v_measure','silhouette_score','calinski_harabasz_score']
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
                              
                        
