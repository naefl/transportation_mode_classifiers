# coding=utf-8
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, Normalizer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import RandomForestClassifier
import inspect
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif

filter = 'butter_5hz_lowpass'

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

data = pd.read_csv('combined_trainingsdata_{}.csv'.format(filter))

data = data[data['speed']!=0][:][1:]
data = data.iloc[:,1:]
print(data.head())
# ================================================transforming========================================================
#all vehicles
X = data.iloc[:,:-1].values
Y = data.iloc[:,-1]
# 0: Bicycle
# 1: Bus
# 2:  Car
# 3: Train
# 4: Tram
# 5: Walking


#bus only
Y_bus = (Y=='bus').values.astype(int)
# 1: bus
# 0: no bus

imputer = Imputer(missing_values='NaN', strategy = 'mean', axis=0)
imputer = imputer.fit(X)
X = imputer.transform(X)

labelencoder_Y =LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
labelencoder_Y_bus = LabelEncoder()
Y_bus = labelencoder_Y_bus.fit_transform(Y_bus)

# ================================================Splitting Training/Test Data==========================================


#training and testing splitting multi class
sc_X =  StandardScaler()
X = sc_X.fit_transform(X)
# ================================================Model Selection=======================================================

#classifier
gb_clf = GradientBoostingClassifier(),
gb_clf_bus = GradientBoostingClassifier(),
svm_clf= SVC(kernel = 'rbf', random_state=0)
svm_clf_bus = SVC(kernel = 'rbf', random_state=0)
xgb_clf = XGBClassifier()
xgb_clf_bus = XGBClassifier()
rf_clf = RandomForestClassifier()
rf_clf_bus = RandomForestClassifier()
clfs = [xgb_clf, xgb_clf_bus
]


# ================================================Cross-Validation======================================================
if __name__ == '__main__':
    accs = []
    accs_bus = []
    reports = []
    reports_bus = []
    conf_mats = []
    conf_mats_bus = []

    for i,clf in enumerate(clfs[::2]):
        print("__Currently ON_____________________________________{}"
                              "_________________________________________________________________________".format(retrieve_name(clf)[0]))
        accs_clf = []
        accs_bus_clf = []
        y_hats = []
        y_bus_hats = []
        for j in range(45):
            print(j)
            X_new=SelectKBest(f_classif,k=j+1).fit(X,Y)
            clf_bus = clfs[1::2][i]
            np.set_printoptions(precision=3)
            print('scores for features non-bus')
            print(X_new.scores_)
            X_new = X_new.transform(X)
            print(X_new)
            y_hats.append(cross_val_predict(clf, X_new,Y,cv=10, n_jobs=-1))
            accs_clf.append(np.average(accuracy_score(Y,y_hats[i])))
            print(accs_clf[-1])
            X_new_bus=SelectKBest(f_classif,k=j+1).fit(X,Y_bus)
            print('scores for features bus')
            print(X_new_bus.scores_)
            X_new_bus = X_new_bus.transform(X)
            print(X_new_bus)
            y_bus_hats.append(cross_val_predict(clf_bus, X_new_bus, Y_bus, cv=10, n_jobs=-1))
            accs_bus_clf.append(accuracy_score(Y_bus,y_bus_hats[i]))
            print(accs_bus_clf[-1])
        accs.append(accs_clf)
        accs_bus.append(accs_bus_clf)


