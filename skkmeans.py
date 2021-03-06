from EvalClus import evaluate
import pandas as pd
import csv

'''
This file runs different configurations of kmeans 
Each parameter different values are defined as paramerter name and “_r” at the end 
All results are written to csv file defined in the code , csv_file = "./output/kmeans_eval_out.csv"
'''

estimator = "kmeans"
config = {"init": "k-means++", "n_clusters": 8, "n_init": 10, "max_iter": 300}

''' nmi is a flag, when it is set to true the model will only evaluate configurations based on ground truth data
'''
nmi = True

if nmi:
      csv_file = "./output/BestModel/kmeans_bestmodel.csv"
else:
      csv_file = "./output/kmeans_eval_out.csv"

count_all=1

n_clusters_r=list(range(2, 40))

count_all*=len(n_clusters_r)

init_r=['k-means++','random']
count_all*=len(init_r)

max_iter_r= [300]#[100, 300, 1000, 5000]
count_all*=len(max_iter_r)

algorithm_r=['auto']#, 'full', 'elkan']
count_all*=len(algorithm_r)

n_init_r=[10]#[2, 10, 50]
count_all*=len(n_init_r)

count=0


for init in init_r:
      for n_clusters in n_clusters_r:
            for max_iter in max_iter_r:
                  for algorithm in algorithm_r:
                        for n_init in n_init_r:
                              config = {"n_clusters": n_clusters, "init": init,
                                    "max_iter": max_iter, "n_init": n_init, "algorithm": algorithm}
                              s = evaluate(estimator, config)
                              s.run_all(verbose=True,nmi=nmi)
                              out = s.res
                              d = {}
                              # for key in out:

                              for dataset in out.keys():
                                    d0 = {"dataset": dataset}
                                    d1 = out[dataset]
                                    d0.update(d1)

                                    d0.update(config)
                                    if nmi: 
                                          dcols=["dataset" , "n_clusters" , "init" , "max_iter" , "n_init" , "algorithm" ,'nmi']
                                    else:
                                          dcols=["dataset" , "n_clusters" , "init" , "max_iter" , "n_init" , "algorithm" ,'baker_hubert_gamma', 'ball_hall', 'ratkowsky_lance', 'davies_bouldin', 'dunns_index', 'mcclain_rao', 'pbm_index', 'ratkowsky_lance', 'ray_turi', 'scott_symons', 'wemmert_gancarski', 'xie_beni', 'c_index', 'det_ratio', 'g_plus_index', 'i_index', 'ksq_detw_index', 'log_det_ratio', 'log_ss_ratio', 'modified_hubert_t', 'point_biserial', 'r_squared', 'root_mean_square',  's_dbw', 'silhouette', 'tau_index', 'trace_w', 'trace_wib', 'iindex', 'sdbw', 'ari', 'ami', 'nmi','v_measure','silhouette_score','calinski_harabasz_score']
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
