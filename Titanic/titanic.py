#!/usr/bin/env python
# -*- coding: utf-8 -*-

# author: mickey0524
# first kaggle competition
# 2017-03-24 

from sklearn.ensemble import RandomForestRegressor
from Titanic import np, pd

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
  X = known_age[: 1:]

  rfr = RandomForestRegressor(random_state = 0, n_estimators = 2000, n_jobs = 1)
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


def fitting_data(df):
  """
  数据预处理
  type df: DataFrame df对象
  rtype: DataFrame
  """
  df = set_missing_ages(df)
  df = set_Cabin_type(df)
  return df


def tiantic(df):
  """
  程序入口
  type df: DataFrame df对象
  """
  data_train = fitting_data(df)
  print data_train


if __name__ == '__main__':
  tiantic(data_train)

