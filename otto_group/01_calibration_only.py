# encoding: utf-8
"""
    https://github.com/christophebourguignat/notebooks/blob/master/Calibration.ipynb
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC, BaggingClassifier as BC
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV

X = pd.read_csv('train.csv')
X= X.drop('id', axis=1)
y = X.target.values
y = LabelEncoder().fit_transform(y)
X = X.drop('target', axis=1)
print X.head(3)

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20, random_state=36)
print 'X splitted'

clf = RFC(n_estimators=250, n_jobs=-1)
# use BaggingClassifier to make 5 predictions, and average
clfbag = BC(clf, n_estimators=5)
print 'fitting bag clf ...'
clfbag.fit(Xtrain, ytrain)
print 'done !'
ypreds = clfbag.predict_proba(Xtest)
# will be 0.60 also
print "%.2f" % log_loss(ytest, ypreds, eps=1e-15, normalize=True)

clf = RFC(n_estimators=250, n_jobs=-1)
# isotonic works better than the default sigmoid in this case
clfcali = CalibratedClassifierCV(clf, method='isotonic', cv=5)
print 'fitting calibration clf ...'
clfcali.fit(Xtrain, ytrain)
print 'done !'
ypreds = clfcali.predict_proba(Xtest)
# will be 0.49 also
print "%.2f" % log_loss(ytest, ypreds, eps=1e-15, normalize=True)
