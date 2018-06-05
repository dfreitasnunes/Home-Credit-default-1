# Este vai ser um projecto para tentar aprender mais sobre data science resolvendo problemas no Kaggle.

# data preprocessing 

# data load 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer 
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
# The path must be accord your data.csv don't forget to change it
Path = "C:\\Users\\Diogo\Desktop\\Machine Learning A-Z\\Home_banking_competition\\"
Dataset = pd.read_csv(Path + "application_train.csv") 

#get rid of missing values and turning categorical variables into numbers
Dataset1 = Dataset
Dataset1 = pd.get_dummies(Dataset1)
Dataset1 = Dataset1.fillna(Dataset1) 

# X independente variables
X =  np.delete(Dataset1.iloc[:,:].values,np.s_[1],1)
# Y dependente variable
Y = Dataset1.iloc[:,1]
