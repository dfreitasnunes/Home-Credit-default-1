# Este vai ser um projecto para tentar aprender mais sobre data science resolvendo problemas no Kaggle.

# Function to get a balanced dataset
def balanced_subsample(x,y,subsample_size=1.0):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs,ys

# librarys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd





# data load and division

Path = "C:\\Users\\Diogo\Desktop\\Machine Learning A-Z\\Home_banking_competition\\"
Dataset = pd.read_csv(Path + "application_train.csv") 

Dataset1 = Dataset

#del Dataset1['SK_ID_CURR']
#del Dataset1['NAME_TYPE_SUITE']
#del Dataset1['OWN_CAR_AGE']

Dataset1 = pd.get_dummies(Dataset1)
Dataset1 = Dataset1.fillna(Dataset1.mean()) 


X =  np.delete(Dataset1.iloc[:,:].values,np.s_[1],1)
Y = Dataset1.iloc[:,1]

# Data Resampling

xs,ys = balanced_subsample(X,Y)


# Data Split for train set and test set

from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(xs,ys, test_size = 0.20 , random_state = 0)

        
# feature scalling 

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# fitting Logist Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,Y_train)

# fitting decision tree
#from sklearn.tree import DecisionTreeClassifier 
#classifier = DecisionTreeClassifier(criterion= 'entropy' , random_state= 0)
#classifier.fit(X_train,Y_train)

# Predition
Y_pred = classifier.predict(X_test)


# confusion matrix
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(Y_test,Y_pred)


#Plots for training




# Plots for test
