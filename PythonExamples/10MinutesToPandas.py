# coding=UTF-8

# coding=UTF-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 创建对象 --------------------------------------------------------------

# 通过传递一个list对象来创建一个Series
s = pd.Series([1, 3, 5, np.nan, 6, 8])
# print(s)

# 通过传递一个numpy array，时间索引以及列标签来创建一个DataFrame
dates = pd.date_range('20130101', periods=6)
# print(dates)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
# print(np.random.randn(6, 4))
# print(df)

# 通过传递一个能够被转换成类似序列结构的字典对象来创建一个DataFrame
df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3] * 4, dtype='int32'),
                    'E': pd.Categorical(["test", "train", "test", "train"]),
                    'F': 'foo'})
# print(df2)

# 查看不同列的数据类型
# print(df2.dtypes)

# --- 查看数据 ---------------------------------------------------------------

# 查看frame中头部和尾部的行
# print(df.head())
# print(df.tail(3))

# 显示索引、列和底层的numpy数据
# print(df.index)
# print(df.columns)
# print(df.values)

# 数据快速统计汇总
# print(df.describe())

# 数据转置
# print(df.T)

# 按轴进行排序
# print(df.sort_index(axis=1, ascending=False))

# 按值进行排序
# print(df.sort_values(by='B'))

# --- 选择 -----------------------------------------------------------------

# 获取

# 选择一个单独的列，这将会返回一个Series，等同于df.A
# print(df['A'])

# 通过[]进行选择，这将会对行进行切片
# print(df[0:3])
# print(df['20130102':'20130104'])

# 通过标签选择

# 使用标签来获取一个交叉的区域
# print(df.loc[dates[0]])

# 通过标签来在多个轴上进行选择
# print(df.loc[:, ['A', 'B']])

# 标签切片
# print(df.loc['20130102':'20130104', ['A', 'B']])

# 对于返回的对象进行维度缩减
# print(df.loc['20130102', ['A', 'B']])

# 获取一个标量
# print(df.loc[dates[0], 'A'])

# 快速访问一个标量（与上一个方法等价）
# print(df.at[dates[0], 'A'])

# 通过位置选择

# 通过传递数值进行位置选择（选择的是行）
# print(df.iloc[3])

# 通过数值进行切片，与numpy/python中的情况类似
# print(df.iloc[3:5, 0:2])

# 通过指定一个位置的列表，与numpy/python中的情况类似
# print(df.iloc[[1, 2, 4], [0, 2]])

# 对行进行切片
# print(df.iloc[1:3, :])

# 对列进行切片
# print(df.iloc[:, 1:3])

# 获取特定的值
# print(df.iloc[1, 1])

# 快速获取特定的值（与上一个方法等价）
# print(df.iat[1, 1])

# 布尔索引

# 使用一个单独列的值来选择数据
# print(df[df.A > 0])

# 使用where操作来选择数据
# print(df[df > 0])

# 使用isin()方法来过滤
df2 = df.copy()
df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
# print(df2)
# print(df2[df2['E'].isin(['two', 'four'])])

# 设置

# 设置一个新的列
s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20130102', periods=6))
# print(s1)
df['F'] = s1
# print(df)

# 通过标签设置新的值
df.at[dates[0], 'A'] = 0
# print(df)

# 通过位置设置新的值
df.iat[0, 1] = 0
# print(df)

# 通过一个numpy数组设置一组新值
df.loc[:, 'D'] = np.array([5] * len(df))
# print(df)

# 通过where操作来设置新的值
df2 = df.copy()
df2[df2 > 0] = -df2
# print(df2)

# --- 缺失值处理 -------------------------------------------------------

# 在pandas中，使用np.nan来代替缺失值，这些值将默认不会包含在计算中

# reindex()方法可以对指定轴上的索引进行改变/增加/删除操作，这将返回原始数据的一个拷贝
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
df1.loc[dates[0]:dates[1], 'E'] = 1
# print(df1)

# 去掉包含缺失值的行
# print(df1.dropna(how='any'))

# 对缺失值进行填充
# print(df1.fillna(value=5))

#  对数据进行布尔填充
# print(pd.isnull(df1))

# --- 相关操作 ---------------------------------------------------------

# 统计

# 执行描述性统计
# print(df.mean())

# 在其他轴上进行相同的操作
# print(df.mean(1))

# 对于拥有不同维度，需要对齐的对象进行操作。Pandas会自动的沿着指定的维度进行广播
s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates).shift(2)
# print(s)
# print(df)
# print(df.sub(s, axis='index'))

# apply

# 对数据应用函数
# print(df)
# print(df.apply(np.cumsum))
# print(df)
# print(df.apply(lambda x: x.max() - x.min()))

# 直方图
s = pd.Series(np.random.randint(0, 7, size=10))
# print(s)
# print(s.value_counts())

# 字符串方法
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
# print(s.str.lower())

# 合并

#
