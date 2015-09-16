References
-----------
[mainpage](https://www.kaggle.com/c/otto-group-product-classification-challenge)

[6-tricks](https://medium.com/@chris_bour/6-tricks-i-learned-from-the-otto-kaggle-challenge-a9299378cd61)
1. Stacking, blending & averaging
2. [Calibration](https://github.com/christophebourguignat/notebooks/blob/master/Calibration.ipynb)
   本 calibration tutorial，未做任何 feature 调整，仅用 CalibratedClassifierCV 替代 BaggingClassifier，就从 0.60 减少到 0.49
   见 01_calibration_only.py，不过它占用了大量(2G 没够用)的内存 (可以调节 n_estimator 等参数来减少内存使用)
3. GridSearchCV and RandomizedSearchCV (picking parameters like: learning_rate, n_estimators, max_depth, subsample, max_features etc.)
4. XGBoost
5. [NN with lasagne](https://github.com/christophebourguignat/notebooks/blob/master/Tuning%20Neural%20Networks.ipynb)
6. Bagging Classifier

[xgboost](https://www.kaggle.com/tqchen/otto-group-product-classification-challenge/understanding-xgboost-model-on-otto-data#script-save-run)

[ensemble-weights](https://www.kaggle.com/hsperr/otto-group-product-classification-challenge/finding-ensamble-weights)
见 02_weighted_ensemble.py，结果为
RFC logloss 0.730902347855
LR logloss 0.672481908536
RFC2 logloss 0.6743768266
Ensemble Score: 0.564591046758
Best weights: [ 0.42321194  0.1531398   0.42364826]
这个脚本的结果并不够好，只是个示例，如果真用起来，需要使用 NN, XGBoost 等代替 RF, LR，并多随机 starting weights，多做尝试

[2nd-guschin](http://blog.kaggle.com/2015/06/09/otto-product-classification-winners-interview-2nd-place-alexander-guschin/)
他主要使用了 Stacking 的方法，基础模型采用 NN, XGBoost, KNN，并着重消除了接近 features 产生的影响

[useful-scripts](http://blog.kaggle.com/2015/06/15/dont-miss-these-scripts-otto-group-product-classification/)


