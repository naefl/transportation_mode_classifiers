from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
from sklearn.ensemble import GradientBoostingClassifier
data = pd.read_csv('combined_trainingsdata_gaussian.csv')

X = pickle.load(open('X_feature_selected.p','rb'))
Y = pickle.load(open('Y.p','rb'))
# ================================================Model Selection=======================================================

n_estimators =np.linspace(200,1000,5, dtype=int)
max_depth = 16
clf_ = GradientBoostingClassifier(
max_depth = max_depth 
)
clf = GridSearchCV(clf_,
            dict(n_estimators=n_estimators
                 ),
                 cv=10,
                 n_jobs=-1)

clf.fit(X, Y)
pickle.dump(clf, open("gridsearch_clf_{}_{}_{}.p".format(n_estimators[0],n_estimators[-1],max_depth),"wb"))


scores = [x[1] for x in clf.grid_scores_]
print(scores)

