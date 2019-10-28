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

bin_seeding_r=[False,False]
count_all*=len(bin_seeding_r)

n_jobs_r=[-1,1,2,3]
count_all*=len(n_jobs_r)

count=0

for cluster_all in cluster_all_r:
      for bin_seeding in bin_seeding_r:
            for n_jobs in n_jobs_r:
 
                config = {"cluster_all": cluster_all, "bin_seeding": bin_seeding,
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

                    dcols=["dataset","silhouette_score","calinski_harabasz_score","davies_bouldin_score","SSE","nSSE","cluster_all","bin_seeding","n_jobs"]
                    with open(csv_file, 'a', newline='') as csvfile:
                            writer = csv.DictWriter(
                                csvfile, delimiter='\t', fieldnames=dcols)
                            writer.writeheader()
                            # for data in dict_data:
                            print(d0.keys())
                            writer.writerow({"dataset": d0["dataset"], "silhouette_score": d0["silhouette_score"], "calinski_harabasz_score": d0["calinski_harabasz_score"], "davies_bouldin_score": d0["davies_bouldin_score"],
                                        "SSE": d0["SSE"], "nSSE": d0["nSSE"], "cluster_all": d0["cluster_all"], "bin_seeding": d0["bin_seeding"], "n_jobs": d0["n_jobs"]})
                        
                count+=1
                print("run "+str(count)+" configs out of "+str(count_all))
                
