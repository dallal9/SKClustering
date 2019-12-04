from EvalClus import evaluate
import pandas as pd
import csv

'''
This file runs different configurations of meanshift 
Each parameter different values are defined as paramerter name and “_r” at the end 
All results are written to csv file defined in the code , csv_file = "./output/meanshift_eval_out.csv"
'''

estimator = "meanshift"
config = {"cluster_all": False, "n_jobs": -1, "bin_seeding": False}
csv_file = "./output/meanshift_eval_out.csv"

count_all=1

cluster_all_r=[False,True]
count_all*=len(cluster_all_r)

bin_seeding_r=[False,True]
count_all*=len(bin_seeding_r)

bandwidth_r = [1,2,3,None]
count_all*=len(bandwidth_r)

n_jobs_r=[-1,1]
count_all*=len(n_jobs_r)

count=0

for cluster_all in cluster_all_r:
      for bin_seeding in bin_seeding_r:
            for n_jobs in n_jobs_r:
                for bandwidth in bandwidth_r:
 
                    config = {"bandwidth":bandwidth,"cluster_all": cluster_all, "bin_seeding": bin_seeding,
                        "n_jobs": n_jobs}
                    s = evaluate(estimator, config)
                    s.run_all()
                    out = s.res
                    d = {}
                    # for key in out:
                    for dataset in out.keys():
                        d0 = {"dataset": dataset}
                        d1 = out[dataset]
                        d0.update(d1)
                        d0.update(config)

                        dcols=["dataset" ,"bandwidth", "cluster_all" , "bin_seeding" , "n_jobs"  ,'Baker_Hubert_Gamma', 'Ball_Hall', 'Banfeld_Raferty', 'Davies_Bouldin', 'Dunns_index', 'McClain_Rao', 'PBM_index', 'Ratkowsky_Lance', 'Ray_Turi', 'Scott_Symons', 'Wemmert_Gancarski', 'Xie_Beni', 'c_index', 'det_ratio', 'g_plus_index', 'i_index', 'ksq_detw_index', 'log_det_ratio', 'log_ss_ratio', 'modified_hubert_t', 'point_biserial', 'r_squared', 'root_mean_square',  's_dbw', 'silhouette', 'tau_index', 'trace_w', 'trace_wib', 'IIndex', 'SDBW', 'ari', 'ami', 'nmi','v_measure']
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
                    
