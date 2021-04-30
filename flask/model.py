import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')
dataset = pd.read_csv('diabetes.csv')
data_X = dataset.iloc[:,[1, 4, 5, 7]].values
data_Y = dataset.iloc[:,8].values
data_X

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
"""
We made use of three alogrithms for predicting if a person is in risk of having diabetes.
First algorithm used is Logistic Regression which had accuray of 78%
Second we made use of KNN which had accuray of 80%
Third we made use of RandomForestClassifier whic gave accuracy of 99%
RandomForestClassifier being the algorithm giving maximum accuracy we processed with this algorithm.
 

"""
models = {"Logistic Regression": LogisticRegression(),
          "KNN": KNeighborsClassifier(),
          "Random Forest": RandomForestClassifier()}          
def fit_and_score(models, X_train, X_test, y_train, y_test):
    np.random.seed(42)
    model_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        model_scores[name] = model.score(X_test, y_test)
    return model_scores
model_scores=fit_and_score(models=models,
                          X_train=X_train,
                          X_test=X_test,
                          y_train=y_train,
                           y_test=y_test)
print(model_scores)

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


