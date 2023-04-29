# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from scipy.spatial import distance
from scipy.stats.stats import pearsonr
from collections import Counter

data_path = 'https://raw.githubusercontent.com/luizfsc/datasets/master/trends_classification.csv'
df_data = pd.read_csv(data_path, header=None)

print(df_data)
X = df_data.loc[:, 0:3].values
labels = df_data.loc[:, 4].values

### NOVOS DADOS 
new_X = np.array([[8, 5, 30, 7], [26, 38, 32, 51]])

new_instance = new_X[0]

### EXTRAINDO AS CORRELACOES (PARTE 1 DO ALGORITMO)

pearson = np.zeros([X.shape[0], 1])
for i in range(X.shape[0]):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
    pearson[i] = pearsonr(X[i, :], new_instance)[0]
print(pearson)

### EXTRAINDO DISTANCIAS (NAO ADEQUADO PARA ESTE PROBLEMA)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist


#(PARTE 2 DO ALGORITMO)
k = 10
nearest_indexes = np.argsort(-pearson.T) 
k_nearest_indexes = nearest_indexes[0][:k]

#print(k_nearest_indexes)

nn_labels = labels[k_nearest_indexes]

#print(nn_labels)

#(PARTE 3 DO ALGORITMO)
# https://docs.python.org/3/tutorial/datastructures.html
classe = 0
count_classes = dict(Counter(nn_labels))
print(count_classes.get(classe))

for i in range(X.shape[0]):
    plt.plot([1, 2, 3, 4], X[i, :], c=('r' if labels[i] == 0 else 'b'))

plt.plot([1, 2, 3, 4], new_instance, c='g')


###-------KNN--------###
def KNN(conjunto_treino, numero_atributos, nova_instancia, valor_k):
    numero_atributos -= 1
    X = conjunto_treino.loc[:, 0:(numero_atributos-1)].values
    labels = conjunto_treino.loc[:, numero_atributos].values
    
    pearson = np.zeros([X.shape[0], 1])
    for i in range(X.shape[0]):
        pearson[i] = pearsonr(X[i, :], nova_instancia)[0]   
    
    k = valor_k
    nearest_indexes = np.argsort(-pearson.T) 
    k_nearest_indexes = nearest_indexes[0][:k]
    nn_labels = labels[k_nearest_indexes]
    
    count_classes = dict(Counter(nn_labels))
    classe_0 = (0 if count_classes.get(0) is None else count_classes.get(0))
    classe_1 = (0 if count_classes.get(1) is None else count_classes.get(1))

    if(classe_0 > classe_1):
        return 0
    elif(classe_1 > classe_0):
        return 1
    else:
        return 

###-------Grafico--------###
for i in range(X.shape[0]):
    plt.plot([1, 2, 3, 4], X[i, :], c=('r' if labels[i] == 0 else 'b'))
plt.plot([1, 2, 3, 4], new_instance, c='g')


print(KNN(df_data, 5, new_instance, 8))
