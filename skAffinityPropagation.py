from EvalClus import evaluate
import pandas as pd
import csv

'''
This file runs different configurations of AffinityPropagation 
Each parameter different values are defined as paramerter name and “_r” at the end 
All results are written to csv file defined in the code , csv_file = "./output/AffinityPropagation_eval_out.csv"
'''

estimator = "AffinityPropagation"
config = {"init": "k-means++", "n_clusters": 8, "n_init": 10, "max_iter": 300}
csv_file = "./output/AffinityPropagation_out.csv"

count_all=1

damping_r=[0.5,0.7,0.8,0.9,1]

count_all*=len(damping_r)

convergence_iter_r=[3,10,15]#, 'random']
count_all*=len(convergence_iter_r)

max_iter_r= [100,200,300]
count_all*=len(max_iter_r)


count=0

for damping in damping_r:
      for convergence_iter in convergence_iter_r:
            for max_iter in max_iter_r:

                config = {"damping": damping, "convergence_iter": convergence_iter,
                    "max_iter": max_iter}
                s = evaluate(estimator, config)
                flag = s.run_all(verbose=True)
                count+=1
                print("run "+str(count)+" configs out of "+str(count_all))
                if flag:
                    out = s.res
                    d = {}
                    # for key in out:

                    for dataset in out.keys():
                            d0 = {"dataset": dataset}
                            d1 = out[dataset]
                            d0.update(d1)

                            d0.update(config)

                            dcols=["dataset" , "damping" , "convergence_iter" , "max_iter" ,'Baker_Hubert_Gamma', 'Ball_Hall', 'Banfeld_Raferty', 'Davies_Bouldin', 'Dunns_index', 'McClain_Rao', 'PBM_index', 'Ratkowsky_Lance', 'Ray_Turi', 'Scott_Symons', 'Wemmert_Gancarski', 'Xie_Beni', 'c_index', 'det_ratio', 'g_plus_index', 'i_index', 'ksq_detw_index', 'log_det_ratio', 'log_ss_ratio', 'modified_hubert_t', 'point_biserial', 'r_squared', 'root_mean_square',  's_dbw', 'silhouette', 'tau_index', 'trace_w', 'trace_wib', 'IIndex', 'SDBW', 'ari', 'ami', 'nmi','v_measure','silhouette_score','calinski_harabasz_score']
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
                
