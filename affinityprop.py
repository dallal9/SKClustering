from EvalClus import evaluate
import pandas as pd
import csv

'''
This file runs different configurations of kmeans 
Each parameter different values are defined as paramerter name and “_r” at the end 
All results are written to csv file defined in the code , csv_file = "./output/kmeans_eval_out.csv"
'''

estimator = "affinityprop"
config = {"affinity": "euclidean", "damping": 0.5,  "max_iter": 300}
csv_file = "./output/affinityprop_eval_out.csv"

count_all=1

affinity_r = ["euclidean", "precomputed"]
damping_r = [0.5, 0.6, 0.7, 0.8, 0.9]
max_iter_r = [300, 500, 1000, 5000]



count=0

for affinity in affinity_r:
      for damping in damping_r:
            for max_iter in max_iter_r:
                  config = {"affinity": affinity, "damping": damping, "max_iter": max_iter}
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
                        print(d0)

                  #      dcols=["dataset","silhouette_score","calinski_harabasz_score","davies_bouldin_score","SSE","nSSE","n_clusters","init","max_iter","n_init","algorithm"]
                  #      with open(csv_file, 'a', newline='') as csvfile:
                  #            writer = csv.DictWriter(
                  #                  csvfile, delimiter='\t', fieldnames=dcols)
                              # for data in dict_data:
                  #            writer.writerow({"dataset": d0["dataset"], "silhouette_score": d0["silhouette_score"], "calinski_harabasz_score": d0["calinski_harabasz_score"], "davies_bouldin_score": d0["davies_bouldin_score"],
                  #                        "SSE": d0["SSE"], "nSSE": d0["nSSE"], "n_clusters": d0["n_clusters"], "init": d0["init"], "max_iter": d0["max_iter"], "n_init": d0["n_init"], "algorithm": d0["algorithm"]})
                  #count+=1
                  #print("run "+str(count)+" configs out of "+str(count_all))
