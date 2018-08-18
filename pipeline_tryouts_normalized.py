# coding=utf-8
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
import numpy as np
from matplotlib.colors import ListedColormap
import math
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, Normalizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_val_predict
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
import time
import inspect
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier

filter = 'butter_5hz_lowpass'

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

data = pd.read_csv('combined_trainingsdata_{}.csv'.format(filter))

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
sc_X =  Normalizer()
X = sc_X.fit_transform(X)
# X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=0)
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)

#training and testing splitting single class
# X_bus_train, X_bus_test, Y_bus_train, Y_bus_test = train_test_split(X,Y_bus, test_size=0.25, random_state=0)
# X_bus_train = sc_X.fit_transform(X_bus_train)

# ================================================Model Selection=======================================================

#classifier

svm_clf= SVC(kernel = 'rbf', random_state=0)
svm_clf_bus = SVC(kernel = 'rbf', random_state=0)
xgb_clf = XGBClassifier()
xgb_clf_bus = XGBClassifier()
dt_clf = DecisionTreeClassifier()
dt_clf_bus = DecisionTreeClassifier()
adb_clf = AdaBoostClassifier()
adb_clf_bus = AdaBoostClassifier()
gNB_clf = GaussianNB()
gNB_clf_bus = GaussianNB()
rf_clf = RandomForestClassifier()
rf_clf_bus = RandomForestClassifier()
knn_clf = KNeighborsClassifier()
knn_clf_bus = KNeighborsClassifier()
logReg_clf = LogisticRegression(random_state=0)
logReg_clf_bus = LogisticRegression(random_state=0)
mlp_clf = MLPClassifier()
mlp_clf_bus = MLPClassifier()
qda_clf = QuadraticDiscriminantAnalysis()
qda_clf_bus = QuadraticDiscriminantAnalysis()
lda_clf = LinearDiscriminantAnalysis()
lda_clf_bus = LinearDiscriminantAnalysis()
gb_clf = GradientBoostingClassifier()
gb_clf_bus = GradientBoostingClassifier()
lsvm_clf = LinearSVC()
lsvm_clf_bus = LinearSVC()


clfs = [ xgb_clf,xgb_clf_bus,
        svm_clf, svm_clf_bus,
        gb_clf, gb_clf_bus,
        mlp_clf, mlp_clf_bus
        ]


# ================================================Cross-Validation======================================================
if __name__ == '__main__':
    y_hats = []
    y_bus_hats = []
    accs = []
    accs_bus = []
    reports = []
    reports_bus = []
    conf_mats = []
    conf_mats_bus = []

    for i,clf in enumerate(clfs[::2]):
        clf_bus = clfs[1::2][i]
        print("currently on {} and {}".format(retrieve_name(clf)[0],retrieve_name(clf_bus)[0]))
        y_hats.append(cross_val_predict(clf, X,Y,cv=10, n_jobs=-1))
        conf_mats.append(pd.DataFrame(
            data= confusion_matrix(Y, y_hats[i]),
            index = ['bicycle','bus','car','train','tram','walking'],
            columns= ['bicycle','bus','car','train','tram','walking']
        ))
        accs.append(np.average(accuracy_score(Y,y_hats[i])))
        reports.append(classification_report(Y, y_hats[i]))
        print("__________________________"
              "______________________{}__________________________"
              "________________________________________".format(retrieve_name(clf)[0]))
        print('multi class 💃')
        print(reports[i])
        print(conf_mats[i])
        print(accs[i])
        print("__________________________________________________________________________________________________________________")
        y_bus_hats.append(cross_val_predict(clf_bus, X, Y_bus, cv=10, n_jobs=-1))
        conf_mats_bus.append(pd.DataFrame(
            data=confusion_matrix(Y_bus, y_bus_hats[i]),
            index=['other vehicle','bus'],
            columns=['other vehicle','bus']
        ))
        accs_bus.append(accuracy_score(Y_bus,y_bus_hats[i]))
        reports_bus.append(classification_report(Y_bus, y_bus_hats[i]))
        print("_________________________________________{}"
              "_________________________________________________________________________".format(retrieve_name(clf_bus)[0]))
        print('single class 🎉')
        print(reports_bus[i])
        print(conf_mats_bus[i])
        print(accs_bus[i])
        print("__________________________________________________________________________________________________________________")

