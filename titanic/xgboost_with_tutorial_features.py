import pandas
import numpy as np
import xgboost as xgb


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

titanic = pandas.read_csv('train.csv')
mean_age = titanic["Age"].median()
mean_fare = titanic["Fare"].median()
data = clean_data(titanic, mean_age, mean_fare)
label = np.array(titanic['Survived'])

dtrain = xgb.DMatrix(data, label=label)
param = {'bst:max_depth': 6, 'bst:eta': 0.1, 'silent': 1, 'objective': 'binary:logistic', 'nthread': 4}
plst = param.items()
plst += [('eval_metric', 'auc')]
plst += [('eval_metric', 'ams@0')]
num_round = 120
# cv = xgb.cv(param, dtrain, num_round, nfold=5, seed=0, metrics={'auc'})
bst = xgb.train(plst, dtrain, num_round)
bst.save_model('xgboost_with_tutorial_features.0001.model')
bst.dump_model('xgboost_with_tutorial_features.0001.txt')

titanic_test = pandas.read_csv("test.csv")
test_data = clean_data(titanic_test, mean_age, mean_fare)
dtest = xgb.DMatrix(test_data)
ypred = bst.predict(dtest)
binary_ypred = map(lambda x: 1 if x > 0.5 else 0, ypred)

submission = pandas.DataFrame({'PassengerId': titanic_test['PassengerId'], 'Survived': binary_ypred})
submission.to_csv('xgboost_with_tutorial_features.0001.csv', index=False)
