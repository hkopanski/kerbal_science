# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 11:33:54 2021

@author: hkopansk
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.chdir(r"C:\Users\hkopansk\Documents\Python Data")

#Read in data (source: https://www.kaggle.com/muhakabartay/sloan-digital-sky-survey-dr16)
df_sky = pd.read_csv("Skyserver_12_30_2019 4_49_58 PM.csv")

#Basic information about the data set
#print(df_sky.head())
#print(df_sky.describe())
#print(df_sky.info())

df_dim_sky = df_sky.shape

df_sky.loc[df_sky['class'] == 'STAR', ['u', 'g', 'r', 'i', 'z']]

df_cnames = list(df_sky)

# Plots
fig, ax = plt.subplots(figsize = (10,6), dpi = 300)

'''sky_colors = {'STAR': 'red', 'GALAXY': 'blue', 'QSO': 'orange'}

ax.scatter(df_sky['g'], 
           df_sky['i'], 
           c = df_sky['class'].map(sky_colors), s = 0.2)
ax.set_xlabel("g")
ax.set_ylabel("i")
ax.grid(True)
plt.legend(fontsize = 5)'''


sky_plot = sns.scatterplot(x = 'r', y = 'z', data = df_sky, hue = 'class', s = 0.75)
plt.grid()
plt.xlim(right = 22)
plt.ylim(top = 22) 
plt.legend(fontsize = 5, loc = 4, markerscale = 0.35)


# KNN classification

np.random.seed(10)
train_values = np.random.choice(range(df_dim_sky[0]), size = int(df_dim_sky[0]*0.5), replace = False)
mask_values = np.isin(np.array(range(df_dim_sky[0])), train_values, invert = True)
test_values = np.array(range(df_dim_sky[0]))[mask_values]


Y_train = df_sky.loc[train_values,'class']
X_train = df_sky.loc[train_values,['u', 'g', 'r', 'i', 'z']]

Y_test = df_sky.loc[test_values,'class']
X_test = df_sky.loc[test_values,['u', 'g', 'r', 'i', 'z']]

k_values = list(range(1, 16))
misclass = []

knn_classifier2 = KNeighborsClassifier()
knn_grid = {'n_neighbors': np.arange(1, 6)}

knn_clssr_grid = GridSearchCV(knn_classifier2, knn_grid, cv = 5)

knn_clssr_grid.fit(X_train, Y_train)

print(knn_clssr_grid.best_params_)

knn_clssr_grid.predict(X_test)
misclass2 = 1 - knn_clssr_grid.score(X_test, Y_test)

print(f'The test MSE is {misclass2}')

'''
for i in k_values:
    knn_classifier = KNeighborsClassifier(n_neighbors = i)
    knn_fit_sky = knn_classifier.fit(X_train, Y_train)
    knn_pred_sky = knn_classifier.predict(X_test)
    misclass.append(1 - sum(np.equal(Y_test, knn_pred_sky)) / len(Y_test))
    #misclass2 = 1 - knn_classifier.score(X_test, Y_test)
'''