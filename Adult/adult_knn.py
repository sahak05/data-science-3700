
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.metrics.cluster import v_measure_score
from sklearn import neighbors

Y_test_labels = []
LENGHT = 1000
MatriceDistanceAdult = []


def initialiserData():
    global MatriceDistanceAdult
    with open('MatriceDistance.csv', 'r', newline='') as file:
        myreader = csv.reader(file, delimiter=',')
        for rows in myreader:
            MatriceDistanceAdult.append(rows)

    global Y_test_labels
    with open('income.csv', 'r', newline='') as file:
        myreader = csv.reader(file, delimiter=',')
        for rows in myreader:
            Y_test_labels.append(rows[0])

    Y_test_labels = np.array(Y_test_labels)
    Y_test_labels = Y_test_labels.astype(int)

    print(Y_test_labels.shape)
    MatriceDistanceAdult = np.array(MatriceDistanceAdult)
    MatriceDistanceAdult = MatriceDistanceAdult.astype(int)
    print(MatriceDistanceAdult.shape)

initialiserData()
print(len(Y_test_labels))
print(len(MatriceDistanceAdult))


def getDistanceAdult(m1, m2):
    return MatriceDistanceAdult[int(m1[0])][int(m2[0])]

features_train = []
features_test = []
features_valid = []


def initialisation():
    global features_train
    for i in range(0, 60):  # 600 vecteurs de données d'entraînement
        features_train.append([i])
    global features_test
    for i in range(60, 80):  # 200 vecteurs de données de test
        features_test.append([i])

    global features_valid
    for i in range(80, 100):  # 200 vecteurs de données de validation
        features_valid.append([i])


initialisation()

labels_train = Y_test_labels[0:60]
labels_test = Y_test_labels[60:80]
labels_valid = Y_test_labels[80:100]


def knn(metric, type=""):
    score_train = []
    score_val = []
    for k in np.arange(1, 50):
        clf = neighbors.KNeighborsClassifier(n_neighbors=k, metric=metric)
        score_train.append(clf.score(features_train, labels_train))
        score_val.append(clf.score(features_valid, labels_valid))

    if type == "Euclidiene":
        plt.plot(np.arange(1, 50), score_train, color='red')
        plt.plot(np.arange(1, 50), score_val, color='blue')
    else:
        plt.plot(np.arange(1, 50), score_train, color='green')
        plt.plot(np.arange(1, 50), score_val, color='yellow')
    plt.show()
    print("la valeur de K = " + str(np.argmax(score_val) + 1))


def v_measure_score_knn(metric):
    clf = neighbors.KNeighborsClassifier(n_neighbors=3, metric=metric)
    fitted = clf.fit(features_train, labels_train)
    predictionKNN = fitted.predict(features_test)

    print("v_measure_score : " + str(v_measure_score(labels_test, predictionKNN)))

knn(getDistanceAdult, "Euclidiene")
v_measure_score_knn(getDistanceAdult)
