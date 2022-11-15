import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.metrics.cluster import v_measure_score
from pyclustering.cluster.kmedoids import kmedoids

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

def k_medoides(matrix, medoids):
    kmedoidsInstance = kmedoids(matrix, medoids, data_type='distance_matrix')
    kmedoidsInstance.process()
    clusters = kmedoidsInstance.get_clusters()

    predictionKMedoide = np.zeros((taille))
    for i in range(0, len(clusters)):
        for j in range(0, len(clusters[i])):
            predictionKMedoide[clusters[i][j]] = i
    
    print("v_measure_score: "+ str(v_measure_score(y_labels, predictionKMedoide)))

list_medoids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

initialisation()
print("Avec la distance Eucludienne")
k_medoides(matriceDistEucludienne, list_medoids)
print("Avec la notion de similarite")
k_medoides(matriceSimilarite, list_medoids)
