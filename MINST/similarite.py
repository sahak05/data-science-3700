#!/usr/bin/env python
# coding: utf-8



import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd #pip install pandas
import random as random
from scipy.spatial import distance

'''
    Don't forget to install modules 
'''

taille = 1000 # on va aller avec les 1000 premieres donnees
vecteur_test = []
vecteurRow_test = []
image_rotate_dataset = []
image_rotate_dataset_rows = []
image_rotate_translate_dataset = []
image_rotate_translate_dataset_rows = []
x_test = []
y_test = []
priors_test = []
counts = [0,0,0,0,0,0,0,0,0,0]

#initialisation
def initialisationData() :
    fichier_test = np.loadtxt(open('mnist.csv', "rb"), delimiter=",",skiprows=1)
    x = fichier_test[:, 1:]
    y = (fichier_test[:, 0]).astype(int)
    y_test.append(y)

    for i in range(len(x)):
        for j in range(len(x[0])):
            if x[i][j] != 0:
                x[i][j] = round(x[i][j] / 255.0)

    global counts
    for i in y:
        counts[i] = counts[i] + 1
    
    global priors_test
    priors_test = [c / len(y) for c in counts]

    print("Liste priors qui nous donne la probabilite d'avoir une image de chaque classe")
    print(priors_test)

    for i in range(0, taille):
        vecteur_test.append(np.reshape(x[i], (28,28)))
    
def rotation(image, angle): #fonction de rotation suivant un angle en dregre. Image 28x28
    gap_x = 14 - int(14 * (np.cos(angle) + np.sin(angle)))
    gap_y = 14 - int( 14 * (np.cos(angle) - np.sin(angle)))

    image_rotate = np.ones((28,28))
    for i in range(0,28):
        for j in range(0,28):
            x_rorate = (int (round (i * np.cos(angle) + j * np.sin(angle)))) + gap_x
            y_rorate = (int (round (-i * np.sin(angle) + j * np.cos(angle)))) + gap_y
            if (x_rorate > -1 and x_rorate < 28) and (y_rorate > -1 and y_rorate < 28):
                image_rotate[i][j] = image[x_rorate][y_rorate]
    
    return image_rotate


# Translation en fonction de la direction
# 0 == pixel en bas |\ 1 == pixel en haut 
# 2 == pixel a gauche \| 3 == pixel a droite \| autre == pas de pixel
def translation(image, direction):

    translation_x = 0
    translation_y = 0
    if direction == 0:
        translation_x = -1
        translation_y = 0
    elif direction == 1:
        translation_x = 1
        translation_y = 0
    elif direction == 2:
        translation_x = 0
        translation_y = 1
    elif direction == 3:
        translation_x = 0
        translation_y = -1
    elif direction == 5:
        translation_x = 2
        translation_y = 0

    image_translate = np.ones((28,28))
    for i in range(0,28):
        for j in range(0,28):
            x_translate = i + translation_x
            y_translate = j + translation_y
            if (x_translate > -1 and x_translate < 28) and (y_translate > -1 and y_translate < 28):
                image_translate[i][j] = image[x_translate][y_translate]
    return image_translate

def imagesInOneDimension():
    for i in range(0, taille):
        vecteurRow_test.append(np.reshape(vecteur_test[i], 784))

# Maintenant Data augmentation
# Premierement rotation des images du jeu de donnees
def setImagesRotate():
    for i in range(0, taille):
        vecteur = []
        for j in range(-4,5):
            k = j * 0.1
            vecteur.append(rotation(vecteur_test[i], k))
        image_rotate_dataset.append(vecteur)
    for i in range(0, taille):
        vecteur = []
        for j in range(0,9):
            vecteur.append(np.array(image_rotate_dataset[i][j]).flatten())
        image_rotate_dataset_rows.append(vecteur)

# Ensuite la translation des images du jeu de donnees
def setImagesTranslate():
    for i in range(0, taille):
        translation1 = []
        for j in range(0, 9):
            translation2 = []
            for k in range(0, 5):
                translation2.append(translation(image_rotate_dataset[i][j],k))
            translation1.append(translation2)
        image_rotate_translate_dataset.append(translation1)
        
    
    for i in range(0, taille):
        translation1=[]
        for j in range(0, 9):
            translation2=[]
            for k in range(0, 5):
                translation2.append(np.array(image_rotate_translate_dataset[i][j][k]).flatten())
            translation1.append(translation2)
        image_rotate_translate_dataset_rows.append(translation1)

# codons maintenant notre notion de similarite

def getCustomDistance(num1, num2, matrix):
    matrix_rot1 = np.argmin( distance.cdist(image_rotate_dataset_rows[num1],[matrix[num2]], 'euclidean') )
    matrix_rot2 = np.argmin( distance.cdist(image_rotate_dataset_rows[num2],[matrix[num1]], 'euclidean') )
    
    matrix_transl1 = distance.cdist(image_rotate_translate_dataset_rows[num1][matrix_rot1],[matrix[num2]], 'euclidean')
    matrix_transl2 = distance.cdist(image_rotate_translate_dataset_rows[num2][matrix_rot2],[matrix[num1]], 'euclidean')
    return 0.5*(min(matrix_transl1)[0] + min(matrix_transl2)[0])

def customDistVsEucludieneDistance(num1, num2):
    print("Ceci est la distance eucludienne: " + str(round(distance.euclidean(vecteurRow_test[num1],vecteurRow_test[num2]), 4)) )
    print("Notion similarite proposee: " + str(round(getCustomDistance(num1, num2, vecteurRow_test), 4)))

def afficherResultatComparaison(tab, y_test):
    print("Comparaison: Distances eculdiennes donnees par notre notion - distance ecludienne normale.")

    for i in range(len(tab)):
        position1, position2 = tab[i]
        print("Test avec le chiffre   "+ str(int(y_test[position1]))  +" et " +  str(int(y_test[position2])) + " respectivement a la position " + str(position1) + " et " +str(position2) + " du test_dataSet ")  
        customDistVsEucludieneDistance(position1, position2)
        print("==================")
        print("==================")

matriceDistEcludienne =  np.zeros((taille, taille))
matriceSimilarite = np.zeros((taille,taille))

def getMatriceDistEcludienne():
    for i in range(0,taille):
        for j in range(i+1, taille):
            matriceDistEcludienne[i,j] = round(distance.euclidean(vecteurRow_test[i],vecteurRow_test[j]), 4)
    
    for j in range(0, taille):
        for i in range(j+1,taille):
            matriceDistEcludienne[i,j] = matriceDistEcludienne[j,i] 

    with open('matriceDistanceEucludienne.csv', 'w', newline='') as file:
        ecriture = csv.writer(file, delimiter=',')
        ecriture.writerows(matriceDistEcludienne)

def getMatriceSimilarite():
    for i in range(0,taille):
        for j in range(i+1, taille):
            matriceSimilarite[i,j] = round(getCustomDistance(i,j,vecteurRow_test), 4)
    
    for j in range(0, taille):
        for i in range(j+1,taille):
            matriceSimilarite[i,j] = matriceSimilarite[j,i] 

    with open('matriceDistanceSimilarite.csv', 'w', newline='') as file:
        ecriture = csv.writer(file, delimiter=',')
        ecriture.writerows(matriceSimilarite)


def randomLine_Test(nombre):
    tab = []
    for i in range(nombre):
        tab.append((random.randrange(taille), random.randrange(taille)))
    
    print("Let\'s print the random lines for test")
    print(tab)

    return tab


def main():
    initialisationData()
    imagesInOneDimension()
    setImagesRotate()
    setImagesTranslate()
    line_test = randomLine_Test(40)
    afficherResultatComparaison(line_test, y_test[0])
    getMatriceDistEcludienne()
    getMatriceSimilarite()

    print("Distance eucludienne position 0: "+str(distance.euclidean(vecteurRow_test[0],vecteurRow_test[0])))
    print("notion de similarite a la position 0: "+str(getCustomDistance(0, 0, vecteurRow_test)))   
    
if __name__ == "__main__":
    main()
