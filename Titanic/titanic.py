#!/usr/bin/env python
# -*- coding: utf-8 -*-

# author: mickey0524
# first kaggle competition
# 2017-03-24 

from sklearn.ensemble import RandomForestRegressor
from Titanic import np, pd
from sklearn import linear_model
import sklearn.preprocessing as preprocessing

from Titanic import data_train


def set_missing_ages(df):
  """
  利用随机森林补齐缺失的age特征
  type df: DataFrame df对象
  rtype: DataFrame
  """
  age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
  known_age = age_df[age_df.Age.notnull()].as_matrix()
  unknown_age = age_df[age_df.Age.isnull()].as_matrix()

  y = known_age[:, 0]
  X = known_age[:, 1:]
  # print known_age
  rfr = RandomForestRegressor(random_state = 0, n_estimators = 2000, n_jobs = -1)
  rfr.fit(X, y)
  predictedAges = rfr.predict(unknown_age[:, 1:])
  df.loc[(df.Age.isnull()), 'Age'] = predictedAges

  return df


def set_Cabin_type(df):
  """
  用于设置Cabin特征为 Yes/No
  type df: DataFrame df对象
  rtype: DataFrame
  """
  df.loc[df.Cabin.notnull(), 'Cabin'] = 'Yes'
  df.loc[df.Cabin.isnull(), 'Cabin'] = 'No'
  return df


def get_dummies_data(df):
  """
  用于将类目型的特征进行特征因子化
  type df: DataFrame df对象
  rtype: DataFrame
  """
  dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
  dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
  dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
  dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')
  df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis = 1)
  df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis = 1, inplace = True)
  return df


def data_preprocessing(df):
  """
  数据预处理
  type df: DataFrame df对象
  rtype: DataFrame
  """
  df = set_missing_ages(df)
  df = set_Cabin_type(df)
  df = get_dummies_data(df)

  scaler = preprocessing.StandardScaler()
  age_scale_param = scaler.fit(df['Age'].as_matrix().reshape(-1, 1))
  df['Age_scaled'] = scaler.fit_transform(df['Age'].as_matrix().reshape(-1, 1), age_scale_param)
  fare_scale_param = scaler.fit(df['Fare'].as_matrix().reshape(-1, 1))
  df['Fare_scaled'] = scaler.fit_transform(df['Fare'].as_matrix().reshape(-1, 1), fare_scale_param)
  return df


def tiantic(df):
  """
  程序入口
  type df: DataFrame df对象
  """
  train_df = data_preprocessing(df)

    # 用正则取出我们要的属性值
  train_df = df.filter(
      regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
  train_np = train_df.as_matrix()

  # y即Survival结果
  y = train_np[:, 0]

  # X即特征属性值
  X = train_np[:, 1:]

  # fit到RandomForestRegressor之中
  clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
  clf.fit(X, y)
  print clf

if __name__ == '__main__':
  tiantic(data_train)

