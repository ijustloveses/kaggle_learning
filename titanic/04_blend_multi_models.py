# encoding=utf-8

"""
    从 ensemble 项目中借鉴 blend-multiple-model 的脚本
    调整了获取数据的部分，以及生成的 submission 使用整数
"""
from __future__ import division
import numpy as np
import pandas as pd
from xgboost_with_tutorial_features import clean_data
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


def logloss(attempt, actual, epsilon=1.0e-15):
    """Logloss, i.e. the score of the bioresponse competition.
    """
    attempt = np.clip(attempt, epsilon, 1.0 - epsilon)
    return - np.mean(actual * np.log(attempt) + (1.0 - actual) * np.log(1.0 - attempt))


if __name__ == '__main__':

    np.random.seed(0)  # seed to shuffle the train set

    n_folds = 10
    verbose = True
    shuffle = False

    # 训练集 X, y 和 测试集 X_submission
    titanic = pd.read_csv('train.csv')
    mean_age = titanic["Age"].median()
    mean_fare = titanic["Fare"].median()
    X = clean_data(titanic, mean_age, mean_fare)
    y = np.array(titanic['Survived'])
    titanic_test = pd.read_csv("test.csv")
    X_submission = clean_data(titanic_test, mean_age, mean_fare)

    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]

    skf = list(StratifiedKFold(y, n_folds))

    clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)]

    print "Creating train and test sets for blending."

    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))

    """
    首先定义结果变量
        dataset_blend_train : 每个训练样本一行，每列对应一个模型的打分
        dataset_blend_test  : 每个测试样本一行，每列对应一个模型的打分

    然后开始对每种模型遍历，每次遍历，填写上面两个变量的一列
    具体的，会对上面的 n_folds 进行 n 次遍历，n_folds 中的 n-1 份用来训练，1 份用于校验
    这样，每种模型其实共计进行了 n 次训练
        每次用 n-1 小份 fit 模型，fit 之后计算 1 小份的结果，这样 n 次后，校验集正好组合成整个 X 训练集
        那么，这个校验集的结果，合在一起就组成了 dataset_blend_train 的对应模型的一列
        对测试机会有不同，每次 fit 之后，计算测试集的结果，那么每个模型显然会得到 n 个结果
        这个 n 个结果放到该模型的临时变量 dataset_blend_test_j 中，这个变量保存每个测试样本每个 n_fold 的结果
        最后，取平均，得到 dataset_blend_test 的对应模型的一列

    两重遍历结束，我们跑完了每个模型的每个 n_fold 的结果，得到了 dataset_blend_train & dataset_blend_test
    我们看到，这两个数据集分别对应每个训练集、测试集样本的每个模型下的一个打分
    我们现在把这些打分作为特征，扔掉原来的真实特征，只是用打分特征，重新跑一个 LR，预测 y
    fit 之后来给测试集打分作为最终结果 ^-^
    """

    for j, clf in enumerate(clfs):
        print j, clf
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            print "Fold", i
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:, 1]
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:, 1]
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)

    print
    print "Blending."
    clf = LogisticRegression()
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict_proba(dataset_blend_test)[:, 1]

    print "Linear stretch of predictions to [0,1]"
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

    print "Saving Results."
    binary_ypred = map(lambda x: 1 if x >= 0.5 else 0, y_submission)
    submission = pd.DataFrame({'PassengerId': titanic_test['PassengerId'], 'Survived': binary_ypred})
    submission.to_csv('04.blend_multi_models.csv', index=False)
