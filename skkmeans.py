from EvalClus import evaluate
import pandas as pd
import csv



estimator = "kmeans"
config = {"init": "k-means++", "n_clusters": 8, "n_init": 10, "max_iter": 300}
csv_file = "./output/kmeans_eval_out_2.csv"

count_all=1

n_clusters_r=range(2, 3)
count_all*=len(n_clusters_r)

init_r=['k-means++', 'random']
count_all*=len(init_r)

max_iter_r= [100, 300, 1000, 5000]
count_all*=len(max_iter_r)

algorithm_r=['auto', 'full', 'elkan']
count_all*=len(algorithm_r)

n_init_r=[2, 10, 50]
count_all*=len(n_init_r)

count=0

# config = {"n_clusters": n_clusters_r[0], "init": init_r[0], "max_iter": max_iter_r[0], "n_init": n_init_r[0], "algorithm": algorithm_r[0]}
# s = evaluate(estimator, config)
# s.run_all()
# out = s.res
# d = {}
# for key in out:
#       for dataset in out.keys():
#             d0 = {"dataset": dataset}
#             d1 = out[dataset]
#             d0.update(d1)

#             d0.update(config)
#             print(d0)

for n_clusters in n_clusters_r:
#      for init in init_r:
#            for max_iter in max_iter_r:
#                  for algorithm in algorithm_r:
#                        for n_init in n_init_r:
                              config = {"n_clusters": n_clusters, "init": init_r[0],
                                    "max_iter": max_iter_r[0], "n_init": n_init_r[0], "algorithm": algorithm_r[0]}
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
            
                              #       dcols=["dataset","silhouette_score","calinski_harabasz_score","davies_bouldin_score","SSE","nSSE","n_clusters","init","max_iter","n_init","algorithm"]
                              #       with open(csv_file, 'a', newline='') as csvfile:
                              #             writer = csv.DictWriter(
                              #                   csvfile, delimiter='\t', fieldnames=dcols)
                              #             # writer.writeheader()
                              #             # for data in dict_data:
                              #             writer.writerow({"dataset": d0["dataset"], "silhouette_score": d0["silhouette_score"], "calinski_harabasz_score": d0["calinski_harabasz_score"], "davies_bouldin_score": d0["davies_bouldin_score"],
                              #                         "SSE": d0["SSE"], "nSSE": d0["nSSE"], "n_clusters": d0["n_clusters"], "init": d0["init"], "max_iter": d0["max_iter"], "n_init": d0["n_init"], "algorithm": d0["algorithm"]})
                              # count+=1
                              # print("run "+str(count)+" configs out of "+str(count_all))
