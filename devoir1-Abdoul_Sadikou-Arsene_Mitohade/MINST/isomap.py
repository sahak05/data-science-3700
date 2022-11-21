import numpy as np
import matplotlib.pyplot as plt
import csv
from numpy.core import numeric
from sklearn.manifold import MDS
from scipy.sparse.csgraph import dijkstra

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
    
    matriceSimilarite = np.array(matriceSimilarite, dtype=numeric)

    with open('matriceDistanceEucludienne.csv', 'r', newline='') as file:
        lecture = csv.reader(file, delimiter=',')
        for row in lecture:
            matriceDistEucludienne.append(row)
    
    matriceDistEucludienne = np.array(matriceDistEucludienne, dtype=numeric)

    test = np.loadtxt(open('mnist.csv',"rb"),delimiter=",",skiprows=1)
    y_labels =test[:,0]
    y_labels = y_labels[:1000]
    y_labels = [int(c) for c in y_labels]

initialisation()


def plot_show(X, title):
    x_min = np.min(X, 0)
    x_max = np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure(figsize=[15,15])
    plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y_labels[i]), color=plt.cm.Set1(y_labels[i]/10.), fontdict={'weight':'bold', 'size':9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()

def matrice_distance_voisins_plus_proches(k_voisins, taille, distance):
    modifyMatrix = []
    for i in range(0,taille):
        modifyMatrix.append([0]*taille)
    for i in range(0, taille):
        index = np.argpartition( np.array(distance[i]), k_voisins)
        for j in range(0,taille):
            if j!=i:
                if j in index[1:k_voisins+1]:
                    modifyMatrix[i][j] = distance[i][j]
                elif j not in index[1:k_voisins+1]:
                    modifyMatrix[i][j] = float('Inf')
    return np.array(modifyMatrix, dtype=float)

def isomap(matrix, title):
    isoMapDist = matrice_distance_voisins_plus_proches(20,taille,matrix)
    isoMapDist = dijkstra(isoMapDist, directed=False)
    mds_resultat = MDS(n_components = 2, dissimilarity='precomputed')
    donnees_transforme = mds_resultat.fit_transform(isoMapDist)

    plot_show(donnees_transforme,title)


isomap(matriceDistEucludienne, "Isomap avec la distance eucludienne")

isomap(matriceSimilarite, "Isomap avec notre notion de similarite")