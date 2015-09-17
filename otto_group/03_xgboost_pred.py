# encoding: utf-8
"""
根据 cv 的结果，来做预测
"""
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

"""
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
param['bst:eta'] = 0.01
param['bst:sumsample'] = 0.7
param['bst:max_depth'] = 10
# param['silent'] = 1
xgmat = xgb.DMatrix(train, label=le.transform(labels))
plst = param.items()
watchlist = []
bst = xgb.train(plst, xgmat, 1000, watchlist)
bst.save_model('xgboost.model')
"""

bst = xgb.Booster({'nthread': 4})
bst.load_model('xgboost.model')
test = pd.read_csv('test.csv')
test.drop(['id'], axis=1, inplace=True)
xgmat_test = xgb.DMatrix(test)
y_out = bst.predict(xgmat_test)
with open('otto_xgboost_0001.csv', 'w') as f:
    f.write("id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n")
    for i in xrange(len(y_out)):
        f.write("%s,%s\n" % (str(i + 1), ','.join([str(s) for s in y_out[i]])))
