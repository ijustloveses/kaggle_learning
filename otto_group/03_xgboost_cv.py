# encoding: utf-8
"""
自己实现的 grid search + cv
其实还可以参考 xgboost/demo/guide-python/sklearn_examples.py，直接使用 GridSearchCV
"""
import itertools
import pandas as pd
import xgboost as xgb
import numpy as np
import sklearn.cross_validation as cv
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder


def estimate_xgboost(X, labels, param, num_round, folds, le):
    kf = cv.KFold(labels.size, n_folds=folds)
    for train_indices, test_indices in kf:
        # 注意，一定要加 values，否则类似 X[train_indices] 是在取列，而不是取行
        # X_train.shape (49502, 93)    X_test.shape (12376, 93)
        X_train, X_test = X.values[train_indices], X.values[test_indices]
        # y_train.shape (49502,)    y_test.shape (12376,)
        y_train, y_test = labels[train_indices], labels[test_indices]
        # y_train[0] & y_test[0] like 'Class_1'，看到是一个字符分类，但是 xgboost 要求是 0,1,2 ...
        # 故此使用 LabelEncoder()
        xgmat = xgb.DMatrix(X_train, label=le.transform(y_train))
        plst = param.items()
        watchlist = []
        # watchlist = [(xgmat, 'train')]
        bst = xgb.train(plst, xgmat, num_round, watchlist)

        xgmat_test = xgb.DMatrix(X_test)
        # y_out.shape  (12376, 9)   对应 9 个分类的概率
        y_out = bst.predict(xgmat_test)
        # 然而 y_test[idx] 是一维的分类字符串，故此无法直接使用 log_loss，那么需要转换一下
        # 外层 map 是对每一个 y_test 元素做映射，于是，x 对应 y_test 中某一个样本的分类结果，如 'Class_1'
        # le.classes_ = ['Class_1', 'Class_2', .... 'Class_9']
        # 内层 map 就是给定 x 的值，根据 x 生成一个数组，数组以 le.classes_ 为蓝本，和 x 一致的元素置 1，否则置 0
        y_test_trans = map(lambda x: map(lambda z: 1 if z == x else 0, le.classes_), y_test)
        y_test_trans = np.array(y_test_trans)
        loss = log_loss(y_test_trans, y_out)
        print("    --> loss: {score}".format(score=loss))


le = LabelEncoder()
train = pd.read_csv('train.csv')
labels = train['target']
train.drop(['target', 'id'], axis=1, inplace=True)
le.fit(labels)

param = {}
param['objective'] = 'multi:softprob'
param['eval_metric'] = 'mlogloss'
param['num_class'] = len(labels.unique())
param['nthread'] = 4
param['silent'] = 1

folds = 5
all_etas = [0.01]
all_subsamples = [0.7, 0.9]
all_depth = [7, 10]
nums_rounds = [120]  # number of boosted trees

# 1 * 2 * 2 * 1 = 4 combinations
for e, s, m, r in list(itertools.product(all_etas, all_subsamples, all_depth, nums_rounds)):
    param['bst:eta'] = e
    param['bst:sumsample'] = s
    param['bst:max_depth'] = m
    print 'e %.3f  --  s %.2f  --  m %d  -- round %d' % (e, s, m, r)
    estimate_xgboost(train, labels, param, r, folds, le)
"""
e 0.010  --  s 0.70  --  m 7  -- round 120
    --> loss: 1.62782689759
    --> loss: 1.63632147529
    --> loss: 1.68480261928
    --> loss: 1.24546674499
    --> loss: 1.94349976446
e 0.010  --  s 0.70  --  m 10  -- round 120
    --> loss: 1.57603631909
    --> loss: 1.59243930154
    --> loss: 1.6674162912
    --> loss: 1.21939557111
    --> loss: 1.90051308171
e 0.010  --  s 0.90  --  m 7  -- round 120
    --> loss: 1.62782689759
    --> loss: 1.63632147529
    --> loss: 1.68480261928
    --> loss: 1.24546674499
    --> loss: 1.94349976446
e 0.010  --  s 0.90  --  m 10  -- round 120
    --> loss: 1.57603631909
    --> loss: 1.59243930154
    --> loss: 1.6674162912
    --> loss: 1.21939557111
    --> loss: 1.90051308171
"""
