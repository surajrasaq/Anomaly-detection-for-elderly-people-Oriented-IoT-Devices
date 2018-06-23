# Author Surajudeen Abdulrasaq

# University of Lyon/University Jean Monnet France

"""
==========================================
Classification with LDA
==========================================

"""

from mlxtend.plotting import plot_decision_regions
import pandas as pd
import numpy as np
import numpy
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_predict, KFold

# Read and Load Data
activity = pd.read_csv('./data/cane_data/pca_dis.csv',  delimiter = ',')
activity1 = pd.read_csv('./data/cane_data/three_act_data.csv', delimiter = ',')



X = np.array(activity.iloc[:,[1,3]])
y = np.array(activity.iloc[:, 4])
y.astype(np.integer)
test = np.array(activity1.iloc[:,[0,3]])
# test.reshape(-1,1)


# Fit the Model
LDA = LinearDiscriminantAnalysis(n_components= 2 )
LDA.fit_transform(X,y)

#plotiing and vissualise
scatter_kwargs = {'s': 120, 'edgecolor': None, 'alpha': 0.7}
contourf_kwargs = {'alpha': 0.2}
scatter_highlight_kwargs = {'s': 120, 'label': 'Test data', 'alpha': 0.7}

plot_decision_regions(X,y, LDA, res= 0.2, legend=5,
                       scatter_kwargs=scatter_kwargs,
                       contourf_kwargs=contourf_kwargs,
                       scatter_highlight_kwargs=scatter_highlight_kwargs
                      )
plt.title('Behavior Classification with Linear Discriminant Analysis')


# remove numbering for proper annotation
#label_dict =  (0,1,2,3,4,5,'Irregular Arm Swing', 'Resting','Vibrating Arm','Exhausted','Normal Walking','Slow Walking')
L = plt.legend(loc = 'best')
L.get_texts()[0].set_text('Sitting')
L.get_texts()[1].set_text('Arm Vibration')
L.get_texts()[2].set_text('Exhaustion')
L.get_texts()[3].set_text('Very Slow walking')
L.get_texts()[4].set_text('Normal Walk')
L.get_texts()[5].set_text('Pausing')

#plt.legend(loc = 'lower right' )
plt.show()

