import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import DistanceMetric
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import math


data_set = pd.read_csv('adult.csv')

"""
    Pour connaitre les details sur les caracteristiques et leurs types
    On releve 14 caracteristiques
"""
# data_set.info()

df = pd.DataFrame(data_set)

def cleanning():
    global df
    global data_set
    df = df.drop(labels=range(5001, 48842), axis=0) # on utilise les 5000 premieres valeurs

    #visualiser la distribution de gender avec le income
    sns.countplot(data=df, x="gender",hue='income')
    plt.show()

    # remplacer tout les ? par NaN
cleanning()
