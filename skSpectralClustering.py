from EvalClus import evaluate
import pandas as pd
import csv

'''
This file runs different configurations of SpectralClustering 
Each parameter different values are defined as paramerter name and “_r” at the end 
All results are written to csv file defined in the code , csv_file = "./output/SpectralClustering_eval_out.csv"
'''

estimator = "SpectralClustering"
config = {"init": "k-means++", "n_clusters": 8, "n_init": 10, "assign_labels": 300}
csv_file = "./output/SpectralClustering_out.csv"


''' nmi is a flag, when it is set to true the model will only evaluate configurations based on ground truth data
'''
nmi = True

if nmi:
      csv_file = "./output/BestModel/SpectralClustering_bestmodel.csv"
else:
      csv_file = "./output/SpectralClustering_out.csv"


count_all=1
n_clusters_r=list(range(2,40))
count_all*=len(n_clusters_r)

eigen_solver_r = [None]#, 'arpack', 'lobpcg', 'amg']
count_all*=len(eigen_solver_r)

affinity_r=['rbf']#,'nearest_neighbors']#, 'random']
count_all*=len(affinity_r)

assign_labels_r= ['kmeans']#,'discretize']
count_all*=len(assign_labels_r)


count=0

for n_clusters in n_clusters_r:
    for eigen_solver in eigen_solver_r:
        for affinity in affinity_r:
                for assign_labels in assign_labels_r:

                    config = {"n_clusters":n_clusters,"eigen_solver": eigen_solver, "affinity": affinity,
                        "assign_labels": assign_labels}
                    s = evaluate(estimator, config)
                    flag = s.run_all(verbose=True,nmi=nmi)
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
                                if nmi:
                                    dcols=["dataset" , "n_clusters" , "eigen_solver" , "affinity","assign_labels" ,'nmi']
                                else:
                                    dcols=["dataset" , "n_clusters" , "eigen_solver" , "affinity","assign_labels" ,'Baker_Hubert_Gamma', 'Ball_Hall', 'Banfeld_Raferty', 'Davies_Bouldin', 'Dunns_index', 'McClain_Rao', 'PBM_index', 'Ratkowsky_Lance', 'Ray_Turi', 'Scott_Symons', 'Wemmert_Gancarski', 'Xie_Beni', 'c_index', 'det_ratio', 'g_plus_index', 'i_index', 'ksq_detw_index', 'log_det_ratio', 'log_ss_ratio', 'modified_hubert_t', 'point_biserial', 'r_squared', 'root_mean_square',  's_dbw', 'silhouette', 'tau_index', 'trace_w', 'trace_wib', 'IIndex', 'SDBW', 'ari', 'ami', 'nmi','v_measure','silhouette_score','calinski_harabasz_score']
                                    
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
                
