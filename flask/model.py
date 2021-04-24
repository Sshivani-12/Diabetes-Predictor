

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


import warnings
warnings.filterwarnings('ignore')

import pickle

dataset = pd.read_csv('diabetes.csv')


data_X = dataset.iloc[:,[1, 4, 5, 7]].values
data_Y = dataset.iloc[:,8].values



data_X


# In[15]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(data_X)

dataset_scaled = pd.DataFrame(dataset_scaled)

X = dataset_scaled
y = data_Y

X


y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42, stratify = dataset['Outcome'] )



models = {"Logistic Regression": LogisticRegression(),
          "KNN": KNeighborsClassifier(),
          "Random Forest": RandomForestClassifier()}

# Create a function to fit and score models
def fit_and_score(models, X_train, X_test, y_train, y_test):
    # Set random seed
    np.random.seed(42)
    # Make a dictionary to keep model scores
    model_scores = {}
    # Loop through models
    for name, model in models.items():
        # Fit the model to the data
        model.fit(X_train, y_train)
        # Evaluate the model and append its score to model_scores
        model_scores[name] = model.score(X_test, y_test)
    return model_scores
# defined function for performing fit and score


model_scores=fit_and_score(models=models,
                          X_train=X_train,
                          X_test=X_test,
                          y_train=y_train,
                           y_test=y_test)
print(model_scores)




"""
from sklearn.svm import SVC
svc = SVC(kernel = 'linear', random_state = 42)
svc.fit(X_train, Y_train)




svc.score(X_test, Y_test)

Y_pred = svc.predict(X_test)
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from sklearn.metrics import classification_report


rf=RandomForestClassifier()
rf.fit(X_train,y_train)
predicted =rf.predict(X_test)


acc =[]
model=[]


x = metrics.accuracy_score(y_test,predicted)
acc.append(x)
model.append("RandomForest")
print("RandomForest accuracy is :",x*100)
print(classification_report(y_test,predicted))




pickle.dump(rf, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))


