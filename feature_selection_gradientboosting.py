# coding=utf-8
import pickle
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

filter = 'butter_5hz_lowpass'


xgb_clf,most_important,most_important_feats,importances_sorted_np,features_sorted,X,Y = pickle.load(open('clf_mostimportantscoresandfeats_importancesandfeats_X_y.p','rb'))
data = pd.read_csv('combined_trainingsdata_{}.csv'.format(filter))

X = data.iloc[:,:-1].values
Y = data.iloc[:,-1]
labelencoder_Y =LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# ================================================Splitting Training/Test Data==========================================
imputer = Imputer(missing_values='NaN', strategy = 'mean', axis=0)
imputer = imputer.fit(X)
X = imputer.transform(X)


#training and testing splitting multi class
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
# ================================================Model Selection=======================================================

#classifier
gb_clf = GradientBoostingClassifier()
gb_clf_bus = GradientBoostingClassifier()
accs=[]
for i in range(17):
    y_hats = (cross_val_predict(gb_clf, data[most_important_feats[:i+1]].values,Y,cv=10,n_jobs=-1))
    acc_clf = (np.average(accuracy_score(Y,y_hats)))
    print(' on {} iteration, accs is'.format(i+1))
    print(acc_clf)
    accs.append(acc_clf)

idx = accs.index(max(accs))+1
max_clf = gb_clf.fit(data[most_important_feats[:idx]].values,Y)

pickle.dump([max_clf,accs,most_important_feats[:idx]],open('max_clf_gb-accs-most_important_feats.p','wb'))

