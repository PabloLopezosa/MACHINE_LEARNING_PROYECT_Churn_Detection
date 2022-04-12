import pandas as pd
import numpy as np
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import pickle

churn = pd.read_csv('C:\\Users\\lopez\\OneDrive\\Escritorio\\BOOTCAMP_DS\\ALUMNO\\3-Machine_Learning\\Entregas\\entrega ML\\churn\\ENTREGABLE\\Data\\Data_procesed\\train.procesed.csv')

#Dividimos el dataset de train
X = churn.drop(['Exited'],1)
y = churn['Exited']

#Hacemos Smooting para balancear los datos
smoting = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors= 10)
X, y = smoting.fit_resample(X, y)

#Prodesamos las X del modelo de entrenamiento
X_train_scaled = StandardScaler().fit_transform(X.values)

#Importamos dataset test
test = pd.read_csv('C:\\Users\\lopez\\OneDrive\\Escritorio\\BOOTCAMP_DS\\ALUMNO\\3-Machine_Learning\\Entregas\\entrega ML\\churn\\ENTREGABLE\\Data\\Data_procesed\\test.procesed.csv')

#Dividimos el dataset de test
X_test = test.drop(['Exited'],1)
y_test = test['Exited']

#Escalamos los datos también en test
X_test_scaled = StandardScaler().fit_transform(X_test.values)


#SVC equilibrado
svc1 = SVC(kernel='poly', degree =3)
svc1.fit(X_train_scaled,y)

#SVC que maximiza la detencion de unos.
svc2 = SVC(kernel='poly', degree =19)
svc2.fit(X_train_scaled,y)

#Mejor modelo que cumple la condicion del 70% de detención de unos.
gbc = GradientBoostingClassifier(n_estimators = 100, random_state=42)
gbc.fit(X_train_scaled, y)


with open('new_model', 'wb') as archivo_salida:
    pickle.dump(svc2, archivo_salida)