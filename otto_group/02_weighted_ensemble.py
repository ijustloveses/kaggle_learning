# encoding: utf-8
"""
    https://www.kaggle.com/hsperr/otto-group-product-classification-challenge/finding-ensamble-weights
"""
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.cross_validation import StratifiedShuffleSplit as SSS
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import log_loss

train = pd.read_csv('train.csv')
labels = train['target']
train.drop(['target', 'id'], axis=1, inplace=True)

# 只生成一组 split (n_iter=1 缺省为 10)
splits = SSS(labels, test_size=0.05, random_state=1234, n_iter=1)
# splits 中是随机划分为两组的 indices，也就是元素的位置号，而不是元素值本身
for train_index, test_index in splits:
    break
train_x, train_y = train.values[train_index], labels.values[train_index]
test_x, test_y = train.values[test_index], labels.values[test_index]

clfs = []
rfc = RFC(n_estimators=50, random_state=4141, n_jobs=-1)
rfc.fit(train_x, train_y)
print('RFC logloss {score}'.format(score=log_loss(test_y, rfc.predict_proba(test_x))))
clfs.append(rfc)

# usually you should use nn and xgboost here, here we use LR & RFC instead

logreg = LR()
logreg.fit(train_x, train_y)
print('LR logloss {score}'.format(score=log_loss(test_y, logreg.predict_proba(test_x))))
clfs.append(logreg)

rfc2 = RFC(n_estimators=50, random_state=1337, n_jobs=-1)
rfc2.fit(train_x, train_y)
print('RFC2 logloss {score}'.format(score=log_loss(test_y, rfc2.predict_proba(test_x))))
clfs.append(rfc2)

### Now finding the optimum weights ###

predictions = []
for clf in clfs:
    predictions.append(clf.predict_proba(test_x))

def log_loss_func(weights):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
        final_prediction += weight * prediction
    return log_loss(test_y, final_prediction)

# algorithm has a starting value, as an example we choose 0.5 for each weight
# In practice, we should choose many random starting weights and run minimize a few times
starting_values = [0.5] * len(predictions)

# adding constraints
cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
# bound weight between 0 and 1
bounds = [(0, 1)] * len(predictions)

# 做优化，更新 weight
res = minimize(log_loss_func, starting_values, method='SLSQP', bounds=bounds, constraints=cons)

print('Ensemble Score: {best_score}'.format(best_score=res['fun']))
print('Best weights: {weights}'.format(weights=res['x']))
