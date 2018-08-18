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

X = data.iloc[:,:-1].values
Y = data.iloc[:,-1]
labelencoder_Y =LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
print('Y shape')
print(Y.shape)
labelencoder_Y =LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# ================================================Splitting Training/Test Data==========================================
imputer = Imputer(missing_values='NaN', strategy = 'mean', axis=0)
imputer = imputer.fit(X)
X = imputer.transform(X)


#training and testing splitting multi class
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
print('X shape before passing to classifier')
print(X.shape)
print('Y shape before passsing to classifier')
print(Y.shape)
# ================================================Model Selection=======================================================

#classifier
xgb_clf = XGBClassifier()
xgb_clf_bus = XGBClassifier()
for i in range(X.shape[1]):
	try:
		X_new=SelectKBest(f_classif,k=i+1).fit(X,Y)
	except ValueError:
		X_new=SelectKBest(f_classif,k='all').fit(X,Y)
	X_new = X_new.transform(X)
	print(X_new)
	y_hats = (cross_val_predict(xgb_clf, X_new,Y,cv=10))
	print(y_hats)
	accs_clf = (np.average(accuracy_score(Y,y_hats)))
	print(accs_clf)
