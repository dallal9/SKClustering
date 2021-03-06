import pandas as pd 
import numpy as np
import csv
from ast import literal_eval
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

df_metafeatures = pd.read_csv("metafeatures.csv")

arrow = []
with open('cvi-results/multi_cvi2.tsv') as tsvfile:
  reader = csv.DictReader(tsvfile, delimiter='\t', fieldnames=None)
  for row in reader:
    arrow.append(row)
  
df = pd.DataFrame(arrow)
df_datasets = df.iloc[:, 0]
df_cvis = df.iloc[:, 1::]
df_cvis = df_cvis.iloc[:,::2]


ols = list()
temp_cvi = []
for idx, row in df_cvis.iterrows():
  for a in range(0, df_cvis.shape[1]):
    temp_cvi = np.asarray(literal_eval(row[a]))
    ols.append([df_datasets.iloc[idx], temp_cvi[0], temp_cvi[1], temp_cvi[2]])

df_new = pd.DataFrame(ols, columns=['dataset', 'cv_1', 'cv_2', 'cv_3'])

mlb = MultiLabelBinarizer()


df = pd.merge(df_new, df_metafeatures, how='left', on='dataset')
df = df.drop_duplicates(subset='dataset')

X = df.drop(['cv_1', 'cv_2', 'cv_3', 'dataset'], axis=1)
X = np.nan_to_num(X)
y = pd.DataFrame(mlb.fit_transform(df[['cv_1','cv_2', 'cv_3']].values), columns=mlb.classes_, index=df.index)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test) 

y_pred = rf_model.predict(X_test) 
#print(y_pred[6])
#print(y_test.values[6])

# # Get Model Accuracy
acc = accuracy_score(y_pred, y_test)
print("% Accuracy: ", acc)