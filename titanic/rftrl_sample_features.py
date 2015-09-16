# encoding=utf-8
"""
效果不好，这个适合大数据，多feature
"""

import rftrl
import pandas
import numpy as np
from numpy import log

def clean_data(titanic, mean_age, mean_fare):
  titanic["Age"] = titanic["Age"].fillna(mean_age)
  titanic["Fare"] = titanic["Fare"].fillna(mean_fare)
  titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
  titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
  titanic['Embarked'] = titanic['Embarked'].fillna('S')
  titanic.loc[titanic['Embarked'] == 'S', 'Embarked'] = 0
  titanic.loc[titanic['Embarked'] == 'C', 'Embarked'] = 1
  titanic.loc[titanic['Embarked'] == 'Q', 'Embarked'] = 2
  predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', "Fare", 'Embarked']
  return np.array(titanic[predictors])

def logloss(act,pred):
  predicted = max(min(pred, 1. - 10e-15), 10e-15)
  return -log(predicted) if act == 1. else -log(1. - predicted)

titanic = pandas.read_csv('train.csv')
mean_age = titanic["Age"].median()
mean_fare = titanic["Fare"].median()
X_train = clean_data(titanic, mean_age, mean_fare)
y_train = np.array(titanic['Survived'])

clf = rftrl.RandomLeaderClassifier(nr_projections=5, random_state=36, l2=1., size_projections=1, verbose=1)
clf2 = rftrl.RandomLeaderClassifier(nr_projections=10, random_state=37, l2=1., size_projections=3, verbose=1)
clf3 = rftrl.RandomLeaderClassifier(nr_projections=100, random_state=37, l2=1., size_projections=3, verbose=1)

clf.project(X_train)
clf2.project(X_train)
clf3.project(X_train)

loss = 0
loss2 = 0
loss3 = 0
loss_ensemble = 0
loss_ensemble_ranked = 0
count = 0
for _ in range(30):
  for e, (x,y) in enumerate(zip(X_train,y_train)):
    if y == 0 or y == 1: # make a binary problem
      count += 1.

      clf.fit(x,e,y)
      pred = clf.predict()
      loss += clf.logloss()
      clf.update(pred)

      clf2.fit(x,e,y)
      pred2 = clf2.predict()
      loss2 += clf2.logloss()
      clf2.update(pred2)

      clf3.fit(x,e,y)
      pred3 = clf3.predict()
      loss3 += clf3.logloss()
      clf3.update(pred3)

      leaders = sorted([(loss/count,pred), (loss2/count,pred2),  (loss3/count,pred3)])
      loss_ensemble_ranked += logloss(y,((leaders[0][1]*3)+(leaders[1][1]*2)+(leaders[2][1]*1))/6.)
      loss_ensemble += logloss(y,(pred+pred2+pred3)/3.)

      print("%f\t%f\t%f\t%s\t%f\t%f\t%f\t\t%f\t%f"%(pred, pred2, pred3, y, loss/count, loss2/count,  loss3/count, loss_ensemble/count, loss_ensemble_ranked/count))
