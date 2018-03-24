#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from Titanic import data_train, pd, DataFrame

fig1 = plt.figure()
fig1.set(alpha=0.2)  # 设定图表颜色alpha参数

plt.subplot2grid((2, 3), (0, 0))             # 在一张大图里分列几个小图
data_train.Survived.value_counts().plot(kind='bar')  # 柱状图
plt.title(u"获救情况 (1为获救)")  # 标题
plt.ylabel(u"人数")

plt.subplot2grid((2, 3), (0, 1))
data_train.Pclass.value_counts().plot(kind="bar")
plt.ylabel(u"人数")
plt.title(u"乘客等级分布（1等级最高）")

plt.subplot2grid((2, 3), (0, 2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel(u"年龄")                         # 设定纵坐标名称
plt.grid(b=True, which='major', axis='y')
plt.title(u"按年龄看获救分布 (1为获救)")


plt.subplot2grid((2, 3), (1, 0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel(u"年龄")  # plots an axis lable
plt.ylabel(u"密度")
plt.title(u"各等级的乘客年龄分布")
# sets our legend for our graph.
plt.legend((u'头等舱', u'2等舱', u'3等舱'), loc='best')


plt.subplot2grid((2, 3), (1, 2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u"各登船口岸上船人数")
plt.ylabel(u"人数")

Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
Survived_Pclass_df = DataFrame({u'获救': Survived_1, u'未获救': Survived_0})
Survived_Pclass_df.plot(kind='bar', stacked=True)
plt.title(u'各乘客等级的获救情况')
plt.xlabel(u'乘客等级')
plt.ylabel(u'人数')

Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
Survived_Sex_df = DataFrame({u'男性': Survived_m, u'女性': Survived_f})
Survived_Sex_df.plot(kind='bar', stacked=True)
plt.title(u'不同性别的获救情况')
plt.xlabel(u'是否获救')
plt.ylabel(u'人数')

Survive_Embarked_0 = data_train.Embarked[data_train.Survived == 0].value_counts(
)
Survive_Embarked_1 = data_train.Embarked[data_train.Survived == 1].value_counts(
)
Survived_Embarked_df = DataFrame(
    {u'获救': Survive_Embarked_1, u'未获救': Survive_Embarked_0})
Survived_Embarked_df.plot(kind='bar', stacked=True)
plt.title(u'不同登船港口的获救情况')
plt.xlabel(u'港口')
plt.ylabel(u'人数')

Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
Survived_Cabin_df = DataFrame({u'有票': Survived_cabin, u'无票': Survived_nocabin})
Survived_Cabin_df.plot(kind='bar')
plt.title(u'有无票的获救情况')
plt.xlabel(u'是否获救')
plt.ylabel(u'人数')

plt.show()


