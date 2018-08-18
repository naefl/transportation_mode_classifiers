from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
from sklearn.ensemble import GradientBoostingClassifier


X = pickle.load(open('X_feature_selected.p','rb'))	
Y = pickle.load(open('Y.p','rb'))
# ================================================Model Selection=======================================================

n_estimators =np.linspace(500,800,12, dtype=int)
clf_ = GradientBoostingClassifier()
clf = GridSearchCV(clf_,
            dict(n_estimators=n_estimators,
                 max_depth=6),
                 cv=10,
                 n_jobs=-1)

clf.fit(X, Y)
pickle.dump(clf, open("gridsearch_clf.p","wb"))


scores = [x[1] for x in clf.grid_scores_]
print(scores)

