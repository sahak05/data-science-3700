import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.metrics.cluster import v_measure_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils.validation import check_array

taille = 1000
matriceDistEucludienne = []
matriceSimilarite = []
y_labels = []

def initialisation():
    
    global matriceDistEucludienne
    global matriceSimilarite
    global y_labels

    with open('matriceDistanceSimilarite.csv', 'r', newline='') as file:
        lecture = csv.reader(file, delimiter=',')
        for row in lecture:
            matriceSimilarite.append(row)
    
    matriceSimilarite = np.array(matriceSimilarite)

    with open('matriceDistanceEucludienne.csv', 'r', newline='') as file:
        lecture = csv.reader(file, delimiter=',')
        for row in lecture:
            matriceDistEucludienne.append(row)
    
    matriceDistEucludienne = np.array(matriceDistEucludienne)

    test = np.loadtxt(open('mnist.csv',"rb"),delimiter=",",skiprows=1)
    y_labels =test[:,0]
    y_labels = y_labels[:1000]

def getPartitionBiniaire(matrix):
    cluster = AgglomerativeClustering(n_clusters=10, affinity='precomputed', linkage='average')
    matrix = np.array(matrix, dtype=float)
    prediction = cluster.fit_predict(matrix)
    return v_measure_score(y_labels, prediction)

initialisation()
print("Partition binaire avec la distance eucludienne")
print("v_measure_score:" + str(getPartitionBiniaire(matriceDistEucludienne)))
print(" \n\n")
print("Partition binaire avec notre similarite")
print("v_measure_score:" + str(getPartitionBiniaire(matriceSimilarite)))