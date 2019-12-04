from time import time
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN 
from sklearn.cluster import MeanShift
from sklearn.impute import SimpleImputer
from sklearn.metrics.cluster import contingency_matrix

import glob
import os 
import pandas as pd
from cvi import validation
import sys
sys.path.insert(0,'other-cvi')
from sdbw import sdbw
from iindex import metric


''' 
evaluate class is used to evaluate differe sklearn clustering algorithms using datasets lcoated in path="./Datasets/processed/"

'''
class evaluate:
      def __init__(self,estimator_label,config,failed_file=False):
            self.estimator_label=estimator_label
            self.config=config           
            self.loaded=self.load_estimator()
            
            self.res={}
            if not failed_file:
                  self.failed=open(estimator_label+"_failed.txt",mode="a")

            self.failed.flush()

      def run_all(self,path="./Datasets/processed/",verbose = False):

            if os.path.exists(path):
                  
                  allFiles = glob.glob(path + "*.csv")
                  count_load=0
                  count_train=0 
                  count_test=0

                  for dfile in allFiles:
                        try:
                              data = pd.read_csv(dfile, header=None,na_values='?')
                              self.y = data.iloc[:,-1]              

                              self.data = data.iloc[:, :-1]
                              filename_w_ext = os.path.basename(dfile)
                              print(filename_w_ext)
                              filename, file_extension = os.path.splitext(filename_w_ext)
                              self.data_label=filename
                              count_load+=1
                              if verbose:
                                    print("loaded "+str(count_load)+" out of "+str(len(allFiles)))
                              if  self.data.isnull().values.any():
                                    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
                                    imp = imp.fit(self.data)
                                    self.data = pd.DataFrame(imp.transform(self.data))
                                  
                        except:
                              print("couldn't load "+dfile)
                                     
                        if self.loaded:
                              try:
                                    self.fit_data()
                              except:
                                    return False
                              count_train+=1
                              if verbose:
                                    print("fitted  "+str(count_load)+" out of "+str(len(allFiles)))
                              #try:
                              Metric= self.eval_metrics()

                              self.res[self.data_label]=Metric

                              count_test+=1
                              if verbose:
                                    print("evaluated "+str(count_load)+" out of "+str(len(allFiles)))
                                          
                              #except:
                              #      print("evaluation problem",self.data_label,self.config)
                              #      self.failed.write(str(self.data_label)+" " +str(self.config))
                              #      self.failed.write("\n")
                              #      self.failed.flush()
                                    
                        else:
                              print("model loading failed")
                              return False
                        


                              
            else:
                  print(path+" doesn't exist")
                  return False
            return True

      def load_estimator(self):
            if self.estimator_label.lower() =="kmeans":
                  self.estimator=KMeans(init=self.config['init'], n_clusters=self.config['n_clusters'], algorithm = self.config["algorithm"]  ,n_init=self.config['n_init'],max_iter=self.config["max_iter"])
                  self.estimator_label="kmeans"
                  return True
            elif self.estimator_label.lower() =="meanshift":
                  self.estimator=MeanShift(cluster_all=self.config["cluster_all"],bin_seeding=self.config["bin_seeding"],n_jobs=self.config["n_jobs"])
                  return True
            elif self.estimator_label.lower() =="dbscan":
                  self.estimator=DBSCAN(leaf_size=self.config["leaf_size"],metric=self.config["metric"],eps=self.config["eps"],min_samples=self.config["min_samples"])
                  return True
            else:
                  print("couldn't load model",self.estimator_label)
                  return False

      def fit_data(self):
            self.estimator.fit(self.data)
      
      def predict_data(self):
            self.estimator.predict(self.data)
      
      def eval_metrics(self):
            sample_size=int(len(self.data)*0.1)
            if sample_size<100:
                  sample_size=len(self.data)
       
            v= validation(np.asmatrix(self.data).astype(np.float), list(self.estimator.labels_))
            
            Metrics = v.run_all()
            try:
                  Ix = metric(self.data, self.estimator.labels_, self.estimator.cluster_centers_)
                  Metrics["IIndex"] =  Ix.IIndex()
            except:
                   Metrics["IIndex"] = "none"
            try:
                  sdbw_c = sdbw(self.data, self.estimator.labels_, self.estimator.cluster_centers_)
                  Metrics["SDBW"] = sdbw_c.sdbw_score()
            except:
                  Metrics["SDBW"] = "none"
            
            Metrics["ari"]=metrics.adjusted_rand_score(self.y ,self.estimator.labels_)
            Metrics["ami"]=metrics.adjusted_mutual_info_score(self.y ,self.estimator.labels_)
            Metrics["nmi"]=metrics.normalized_mutual_info_score(self.y ,self.estimator.labels_)
            Metrics["v_measure"]=metrics.v_measure_score(self.y, self.estimator.labels_)
            Metrics["silhouette_score"] = metrics.silhouette_score(self.data, self.estimator.labels_, metric='euclidean', sample_size=sample_size,random_state=0)
            Metrics["calinski_harabasz_score"]= metrics.calinski_harabasz_score(self.data,  self.estimator.labels_) 


            '''
            sample_size=int(len(self.data)*0.1)
            if sample_size<100:
                  sample_size=len(self.data)
            Metrics={}
            Metrics["silhouette_score"] = metrics.silhouette_score(self.data, self.estimator.labels_, metric='euclidean', sample_size=sample_size,random_state=0)
            Metrics["calinski_harabasz_score"]= metrics.calinski_harabasz_score(self.data,  self.estimator.labels_) 
            Metrics["davies_bouldin_score"]=metrics.davies_bouldin_score(self.data,  self.estimator.labels_) 
            if self.estimator_label.lower()=="meanshift":
                   Metrics["SSE"]=len(self.estimator.cluster_centers_)

            if self.estimator_label.lower()=="kmeans":
                  araujo = metric(self.data, self.estimator.labels_, self.estimator.cluster_centers_)
                  Metrics["IIndex"] = 0# araujo.IIndex()
                  Metrics["SSE"]=self.estimator.inertia_
                  Metrics["nSSE"]=self.estimator.inertia_/(len(self.data)*len(self.data.columns))

                  labels_true=self.y

                  labels_true=np.array(labels_true)
                  Metrics["ARI"]=metrics.adjusted_rand_score(labels_true, self.estimator.labels_)  
                  Metrics["MIS"]=metrics.adjusted_mutual_info_score(labels_true, self.estimator.labels_)
                  Metrics["v_measure"]=metrics.v_measure_score(labels_true, self.estimator.labels_)

            else:
                  #Metrics["SSE"] = -1
                  Metrics["nSSE"] = -1
            '''

            return Metrics
