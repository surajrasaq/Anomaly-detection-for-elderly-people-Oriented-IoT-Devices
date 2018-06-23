# Author Surajudeen Abdulrasaq

# University of Lyon/University Jean Monnet France

"""
==========================================
PCA for dimension treatment
==========================================

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import csv

activity = pd.read_csv('./evaluate/thirtydays_final.csv',  delimiter = ',')

#activity1 = pd.read_csv('./datas/new_novins.csv',  delimiter = ',')

#pd.to_datetime(activity['date'])
activity.dropna(inplace= True)
del activity['date']

# convert to standard form

s = StandardScaler().fit_transform(activity)
normalize = pd.DataFrame(data = s)
print(normalize.head())

#do the PCA
pca = PCA(n_components=3)
prin_Comp = pca.fit_transform(s)
prin_CompDf = pd.DataFrame(data = prin_Comp
             , columns = ['prin_comp1', 'prin_comp2', 'prin_comp3'])

prin_CompDf.head()

# Join the label to the data and un-comment 'for label data below'
# pca_data = pd.concat([prin_CompDf, activity[['0']]], axis = 1)
# print(pca_data.head(5))

# for no-label data
# normalize.to_csv('./datas/normalize.csv')
prin_CompDf.to_csv('./evaluate/thirtydays_feature.csv')
plt.semilogy(prin_CompDf, '--o')
plt.title('Feature after PCA')
plt.show()

# For label data
# # normalize.to_csv('./datas/normalize.csv')
# pca_data.to_csv('./data/cane_data/pca_dis.csv')
# plt.semilogy(pca_data, '--o')
# plt.title('Feature after PCA')
# plt.show()


