import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import math


data_set = pd.read_csv('adult.csv')

"""
    Pour connaitre les details sur les caracteristiques et leurs types
    On releve 14 caracteristiques
"""
#data_set.info()

df = pd.DataFrame(data_set)

def cleanning():
    global df
    global data_set
    df = df.drop(labels=range(500, 48842), axis=0) # on utilise les 500 premieres valeurs

    #visualiser la distribution de gender avec le income
    sns.countplot(data=df, x="gender",hue='income')
    plt.show()

    # remplacer tout les ? par NaN
    df['native-country'] = df['native-country'].replace('?', np.nan)
    df['workclass'] = df['workclass'].replace('?', np.nan)
    df['occupation'] = df['occupation'].replace('?', np.nan)
    df.dropna(how='any', inplace=True)

    #enlever les colonnes non essentielles
    df.drop(['age', 'relationship', 'educational-num', 'hours-per-week', 'fnlwgt', 'capital-gain', 'capital-loss', 'native-country'], axis=1, inplace=True)

    # column income avec des 0 et 1
    df['income'] = df['income'].map({'<=50K': 0, '>50K': 1}).astype(int)

    df['income'].to_csv('income.csv', index=False, header=False)

    df['gender'] = df['gender'].map({'Male':0.75 , 'Female': 0.2}).astype(float)
    df['race'] = df['race'].map({'Black':0.5 , 'Asian-Pac-Islander': 0.75, 'Other': 0.5, 'White': 0.75, 'Amer-Indian-Eskimo': 0.4}).astype(float)
    df['marital-status'] = df['marital-status'].map({'Married-spouse-absent': 0.75, 'Widowed': 0.50, 'Married-civ-spouse': 0.75, 'Separated': 0.2, 'Divorced': 0.4,
         'Never-married': 0.2, 'Married-AF-spouse': 0.75}).astype(float)
    df['workclass'] = df['workclass'].map({'Self-emp-inc': 0.75, 'State-gov': 0.4, 'Federal-gov': 0.5, 'Without-pay': 0.2, 'Local-gov': 0.4, 'Private': 0.2, 'Self-emp-not-inc': 0.75}).astype(float)
    df['education'] = df['education'].map(
        {'Some-college': 0.5, 'Preschool': 0.2, '5th-6th': 0.2, 'HS-grad': 0.4, 'Masters': 0.75, '12th': 0.4,
         '7th-8th': 0.2,
         'Prof-school': 0.75, '1st-4th': 0.2, 'Assoc-acdm': 0.5, 'Doctorate': 0.75, '11th': 0.4, 'Bachelors': 0.75,
         '10th': 0.4,
         'Assoc-voc': 0.5, '9th': 0.2}).astype(float)
    df['occupation'] = df['occupation'].map(
        {'Farming-fishing': 0.2, 'Tech-support': 0.4, 'Adm-clerical': 0.2, 'Handlers-cleaners': 0.2,
         'Prof-specialty': 0.75, 'Machine-op-inspct': 0.5, 'Exec-managerial': 0.75, 'Priv-house-serv': 0.5,
         'Craft-repair': 0.75, 'Sales': 0.75, 'Transport-moving': 0.5, 'Armed-Forces': 0.75, 'Other-service': 0.2, 'Protective-serv': 0.75}).astype(float)
    # juste pour mieux visualiser notre tableau dans le terminal
    pd.set_option('display.width', 400)
    pd.set_option('display.max_columns', 10)
    
    print(df)

def similarite():
    #score individuel sur tout le df
    scoreIndividuel = []
    for i in df.transpose():
        score = 0 
        for j in df.transpose()[i]:
            score+=j
        scoreIndividuel.append(round(score, 1))
    
    print(scoreIndividuel)
    
    distBetweenIndividual = []
    

    for i in range(len(scoreIndividuel)):
        distBetweenOneAndAll = []
        for j in range(len(scoreIndividuel)):
            if scoreIndividuel[i] >= 5.0:
                distBetweenOneAndAll.append(1)
            else:
                distBetweenOneAndAll.append(0)
            distBetweenIndividual.append(distBetweenOneAndAll)
    #print(distBetweenIndividual)

    # exportons la matrice de distance dans un fichier csv
    with open('MatriceDistance.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(distBetweenIndividual)
    

cleanning() 
similarite()