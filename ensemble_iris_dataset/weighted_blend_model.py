# encoding: utf-8
"""
    http://sebastianraschka.com/Articles/2014_ensemble_classifier.html
"""
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.pipeline import _name_estimators
import numpy as np


class EnsembleClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, clfs, voting='hard', weights=None):
        """
            voting: if 'hard', uses predicted class labels for majority rule voting
                    if 'soft', predicts the class label based on the argmax of the sums of the predicted probalities
        """
        self.clfs = clfs
        self.named_clfs = {key:value for key, value in _name_estimators(clfs)}
        self.voting = voting
        self.weights = weights

    def fit(self, X, y):
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output classification is not supported')
        self.le_ = LabelEncoder()
        self.le_.fit(y)
        self.classes_ = self.le_.classes_
        self.clfs_ = []
        for clf in self.clfs:
            fitted_clf = clone(clf).fit(X, self.le_.transform(y))
            self.clfs_.append(fitted_clf)
        return self

    def predict(self, X):
        if self.voting == 'soft':
            maj = np.argmax(self.predict_proba(X), axis=1)
        else:
            predictions = self._predict(X)
            # bincount like: np.argmax(np.bincount([1, 2, 2], weights=[3, 1, 1]))
            # 就是创建从 0 开始的桶，输入 [1, 2, 2] 分别对应 1号桶和2号桶，分别为一次和两次
            # 对应次的权重为 3，1，1，那么就是说 1号桶1次，权重3； 2号桶2次，权重1
            # 返回 1，指1号桶的加权次数最大
            # 分解来看，np.bincount([1, 2, 2], weights=[3, 1, 1]) = [0, 3, 2] 对应 0，1，2 号桶的加权次数
            maj = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                                      axis=1, arr=predictions)
        maj = self.le_.inverse_transform(maj)
        return maj

    def predict_proba(self, X):
        avg = np.average(self._predict_probas(X), axis=0, weights=self.weights)
        return avg

    def _predict_probas(self, X):
        return np.asarray([clf.predict_proba(X) for clf in self.clfs_])

    def _predict(self, X):
        return np.asarray([clf.predict(X) for clf in self.clfs_]).T

    def transform(self, X):
        if self.voting == 'soft':
            return self._predict_probas(X)
        else:
            return self._predict(X)

    def get_params(self, deep=True):
        if not deep:
            return super(EnsembleClassifier, self).get_params(deep=False)
        else:
            out = self.named_clfs.copy()
            for name, step in six.iteritems(self.named_clfs):
                for k, v in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, k)] = v
            return out


from sklearn import datasets
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target
np.random.seed(123)

clf1 = LogisticRegression()
clf2 = RandomForestClassifier()
clf3 = GaussianNB()
eclf = EnsembleClassifier(clfs=[clf1, clf2, clf3], voting='hard')

for clf, label in zip([clf1, clf2, clf3, eclf], ['LR', 'RF', 'NB', 'Ensemble']):
    scores = cross_validation.cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print "Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label)

# ensemble of ensemble
eclf1 = EnsembleClassifier(clfs=[clf1, clf2, clf3], voting='soft', weights=[5,2,1])
eclf2 = EnsembleClassifier(clfs=[clf1, clf2, clf3], voting='soft', weights=[4,2,1])
eclf3 = EnsembleClassifier(clfs=[clf1, clf2, clf3], voting='soft', weights=[1,2,4])
eclf = EnsembleClassifier(clfs=[eclf1, eclf2, eclf3], voting='soft', weights=[2,1,1])
scores = cross_validation.cross_val_score(eclf, X, y, cv=5, scoring='accuracy')
print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), 'Ensemble of ensemble'))

# grid search
from sklearn.grid_search import GridSearchCV

clf1 = LogisticRegression()
clf2 = RandomForestClassifier()
clf3 = GaussianNB()
eclf = EnsembleClassifier(clfs=[clf1, clf2, clf3], voting='soft')
params = {'logisticregression__C': [1.0, 100.0],
          'randomforestclassifier__n_estimators': [20, 200]}
grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5)
grid.fit(iris.data, iris.target)
for params, mean_score, scores in grid.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params))

"""
output like:
Accuracy: 0.90 (+/- 0.05) [LR]
Accuracy: 0.92 (+/- 0.05) [RF]
Accuracy: 0.91 (+/- 0.04) [NB]
Accuracy: 0.95 (+/- 0.05) [Ensemble]
Accuracy: 0.95 (+/- 0.03) [Ensemble of ensemble]
0.960 (+/-0.012) for {'randomforestclassifier__n_estimators': 20, 'logisticregression__C': 1.0}
0.960 (+/-0.012) for {'randomforestclassifier__n_estimators': 200, 'logisticregression__C': 1.0}
0.960 (+/-0.012) for {'randomforestclassifier__n_estimators': 20, 'logisticregression__C': 100.0}
0.960 (+/-0.012) for {'randomforestclassifier__n_estimators': 200, 'logisticregression__C': 100.0}
"""
