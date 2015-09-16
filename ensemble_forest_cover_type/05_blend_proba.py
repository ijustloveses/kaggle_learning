# encoding: utf-8
"""
    https://github.com/MLWave/Kaggle-Ensemble-Guide/blob/master/blend_proba.py
    这个脚本使用一个传入的模型，n_folds 来做训练，然后保存训练样本在不同结果分类中得到的概率值
    并计算 n_folds 的平均 loss，最后保存模型参数和 n_folds 得到的训练样本结果矩阵
    对不同模型运行脚本并保存文件，我们可以使用这些结果做为新的特征，并使用平均 loss 作为权重，
    进行新的 ensemble 训练
    这个脚本其实就是 04 脚本的内层循环部分单独拉出来，并做了计算平均权重和保存模型
"""
from sklearn import cross_validation
from sklearn.metrics import log_loss, accuracy_score
import numpy as np


def blend_proba(clf, X_train, y, X_test, nfolds=5, save_preds="", save_test_only="",
                seed=300373, save_params="", clf_name="XX", generalizers_params=[],
                minimal_loss=0, return_score=False, minimizer="log_loss"):
    folds = list(cross_validation.StratifiedKFold(y, nfolds, shuffle=True, random_state=seed))
    print X_train.shape
    # 看到，每行一个训练样本，每列则对应分类中的一种，于是每行保存样本对应每个分类的预测
    dataset_blend_train = np.zeros((X_train.shape[0], np.unique(y).shape[0]))

    # 下面开始训练
    loss = 0
    for i, (train_index, test_index) in enumerate(folds):
        # 每个 fold 做一次 fit
        print "Training Fold %s/%s ..." % (i+1, nfolds)
        fold_X_train = X_train[train_index]
        fold_y_train = y[train_index]
        fold_X_test = X_train[test_index]
        fold_y_test = y[test_index]
        clf.fit(fold_X_train, fold_y_train)
        # 然后对 fold_X_test 做一次 predict
        fold_preds = clf.predict_proba(fold_X_test)
        fold_preds_a = np.argmax(fold_preds, axis=1)  # 找到一行中分类预测最大的那个，作为解
        # 计算 loss
        lloss = log_loss(fold_y_test, fold_preds)
        aloss = accuracy_score(fold_y_test, fold_preds_a)
        print "Logistic Loss: %s" % lloss
        print "Accuracy: %s" % aloss
        # NOTE: 从这里看到，这个 clf 模型预测输出的结果，似乎是一个向量，为每个分类的概率
        # 那么，同样，y 是一个矩阵，每一行是一个训练样本，每行为该样本的分类向量，只有一个 1， 其他都是 0
        dataset_blend_train[test_index] = fold_preds
        if minimizer == 'log_loss':
            loss += lloss
        if minimizer == 'accuracy':
            loss += aloss
        # 如果参数中设置了 loss 的最小阈值，而且第一个 fold 的 loss 真的超过了阈值，那么就退出
        if minimal_loss > 0 and loss > minimal_loss and i == 0:
            return False, False
        fold_preds = fold_preds_a
    avg_loss = loss / float(i + 1)  # n 次 folds 的平均 loss
    print "Average loss: %s\n" % avg_loss
    # n folds 遍历训练完毕，再用模型 fit 全部训练样本
    clf.fit(X_train, y)
    # 并做出预测
    # dataset_blend_test = clf.predict_proba(X_test)

    if clf_name == "XX":
        clf_name = str(clf)[1: 3]
    # 此时可以根据 save_xx 参数，来对 dataset_blend_train & dataset_blend_test & clf 的参数等
    # 做一些保存文件工作，保存的方法可以使用 np.save，从略
