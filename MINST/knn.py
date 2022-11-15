import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.metrics.cluster import v_measure_score
from sklearn import neighbors

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
    
    matriceSimilarite = np.array(matriceSimilarite, dtype=float)

    with open('matriceDistanceEucludienne.csv', 'r', newline='') as file:
        lecture = csv.reader(file, delimiter=',')
        for row in lecture:
            matriceDistEucludienne.append(row)
    
    matriceDistEucludienne = np.array(matriceDistEucludienne, dtype=float)

    test = np.loadtxt(open('mnist.csv',"rb"),delimiter=",",skiprows=1)
    y_labels =test[:,0]
    y_labels = y_labels[:1000]

initialisation()

"""
    Separer le jeu de donnees => Entrainement Validation Test
                                    600         200       200
                                    60%         20%        20% 
"""

#Entrainement
labels_train = y_labels[0:600]
#test
labels_test = y_labels[600:800]
#validation
labels_validation = y_labels[800:1000]



train = []
test = []
validation = []
for i in range(0,600):
    train.append([i])

for i in range(600,800):
    test.append([i])

for i in range(800, 1000):
    validation.append([i])

def knn(metric, type=""):
    score_train = []
    score_validation = []

    for k in np.arange(1, 70):
        classif = neighbors.KNeighborsClassifier(n_neighbors=k, metric=metric)
        classif.fit(train, labels_train)
        score_train.append(classif.score(train, labels_train))
        score_validation.append(classif.score(validation, labels_validation))

    if type == "Similarite":
        plt.plot(np.arange(1, 70), score_train, color='green')
        plt.plot(np.arange(1, 70), score_validation, color='black')
        plt.show()
    else:
        plt.plot(np.arange(1, 70), score_train, color='red')
        plt.plot(np.arange(1, 70), score_validation, color='blue')
        plt.show()
    best_k = np.argmax(score_validation) + 1

    print("BEST K = " + str(best_k))

def v_measure_score_knn(metric):
    classif = neighbors.KNeighborsClassifier(n_neighbors=20, metric=metric)
    fit = classif.fit(train, labels_train)
    predictionKNN = fit.predict(test)
    print("Le score est:" + str(v_measure_score(labels_test, predictionKNN)))


def getDistanceEuclidiene(m1,m2):
     return matriceDistEucludienne[int(m1[0])][int(m2[0])]

def getDistanceSimilarite(m1,m2):
     return matriceSimilarite[int(m1[0])][int(m2[0])]



print("KNN avec la distance eucludienne")
knn(getDistanceEuclidiene)
v_measure_score_knn(getDistanceEuclidiene)
print("\n\n\n")
print("KNN avec notre notion de similarite")
knn(getDistanceSimilarite, "Similarite")
v_measure_score_knn(getDistanceSimilarite)