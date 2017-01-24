# coding=UTF-8

# refer to:
# （1）翻译：http://www.cnblogs.com/chaosimple/p/4153083.html
# （2）原文：http://pandas.pydata.org/pandas-docs/stable/10min.html

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

# --- 合并 -----------------------------------------------------------------

# Concat
df = pd.DataFrame(np.random.randn(10, 4))
# print(df)
pieces = [df[:3], df[3:7], df[7:]]
# print(pd.concat(pieces))

# Join 类似于SQL类型的合并
left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
# print(left)
# print(right)
# print(pd.merge(left, right, on='key'))
left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})
# print(left)
# print(right)
# print(pd.merge(left, right, on='key'))

# Append 将一行连接到一个DataFrame上
df = pd.DataFrame(np.random.randn(8, 4), columns=['A', 'B', 'C', 'D'])
# print(df)
s = df.iloc[3]
# print(s)
# print(df.append(s, ignore_index=True))

# --- 分组 ------------------------------------------------------------------

# 对于”group by”操作，我们通常是指以下一个或多个操作步骤：
# · （Splitting）按照一些规则将数据分为不同的组
# · （Applying）对于每组数据分别执行一个函数
# · （Combining）将结果组合到一个数据结构中
df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar',
                          'foo', 'bar', 'foo', 'foo'],
                   'B': ['one', 'one', 'two', 'three',
                          'two', 'two', 'one', 'three'],
                   'C': np.random.randn(8),
                   'D': np.random.randn(8)})
# print(df)

# 分组并对每个分组执行sum函数
# print(df.groupby('A').sum())

# 通过多个列进行分组形成一个层次索引，然后执行函数
# print(df.groupby(['A', 'B']).sum())

# --- Reshaping -----------------------------------------------------------

# Stack
tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
                     'foo', 'foo', 'qux', 'qux'],
                    ['one', 'two', 'one', 'two',
                     'one', 'two', 'one', 'two']]))
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
# print(index)
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
# print('-- df --')
# print(df)
df2 = df[:4]
# print('-- df2 --')
# print(df2)
stacked = df2.stack()
# print('-- stacked --')
# print(stacked)
# print('-- stacked.unstack() --')
# print(stacked.unstack())
# print('-- stacked.unstack(1) --')
# print(stacked.unstack(1))
# print('-- stacked.unstack(0) --')
# print(stacked.unstack(0))

# 数据透视表
df = pd.DataFrame({'A': ['one', 'one', 'two', 'three'] * 3,
                   'B': ['A', 'B', 'C'] * 4,
                   'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
                   'D': np.random.randn(12),
                   'E': np.random.randn(12)})
# print(df)
# print(pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C']))

# --- 时间序列 ----------------------------------------------------------

rng = pd.date_range('1/1/2012', periods=100, freq='S')
# print(rng)
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
# print(ts)
# print(ts.resample('5Min').sum())

# 时区表示
rng = pd.date_range('3/6/2012 00:00', periods=5, freq='D')
ts = pd.Series(np.random.randn(len(rng)), rng)
# print(ts)
ts_utc = ts.tz_localize('UTC')
# print(ts_utc)

# 时区转换
# print(ts_utc.tz_convert('US/Eastern'))

# 时间跨度转换
rng = pd.date_range('1/1/2012', periods=5, freq='M')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
# print(ts)
ps = ts.to_period()
# print(ps)
# print(ps.to_timestamp())

# 时期和时间戳之间的转换使得可以使用一些方便的算术函数
prng = pd.period_range('1990Q1', '2000Q4', freq='Q-NOV')
# print(prng)
ts = pd.Series(np.random.randn(len(prng)), prng)
ts.index = (prng.asfreq('M', 'e') + 1).asfreq('H', 's') + 9
# print(ts.head())

# --- Categoricals -----------------------------------------------

df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "raw_grade": ['a', 'b', 'b', 'a', 'a', 'e']})
# print(df)

# 将原始的grade转换为Categorical数据类型
df["grade"] = df["raw_grade"].astype("category")
# print(df)
# print(df["grade"])

# 将Categorical类型数据重命名为更有意义的名称
df["grade"].cat.categories = ["very good", "good", "very bad"]
# print(df)

# 对类别进行重新排序，增加缺失的类别
df["grade"] = df["grade"].cat.set_categories(["very bad", "bad", "medium", "good", "very good"])
# print(df)
# print(df["grade"])

# 排序是按照Categorical的顺序进行的而不是按照字典顺序进行
# print(df.sort_values(by="grade"))

# 对Categorical列进行排序时存在空的类别
# print(df.groupby("grade").size())

# --- 画图 -------------------------------------------------------------------

ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
# print(ts)
ts = ts.cumsum()
# print(ts)
# ts.plot()
# plt.show()

df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index,
                  columns=['A', 'B', 'C', 'D'])
df = df.cumsum()
# df.plot()
# plt.legend(loc='best')
# plt.show()

# --- 导入和保存数据 ----------------------------------------------------------

# 写入csv文件
# df.to_csv('foo.csv')
# 从csv文件中读取
# print(pd.read_csv('foo.csv'))

# 写入HDF5存储
# df.to_hdf('foo.h5', 'df')
# 从HDF5存储中读取
# print(pd.read_hdf('foo.h5', 'df'))

#  写入excel文件
# df.to_excel('foo.xlsx', sheet_name='Sheet1')
# 从excel文件中读取
# print(pd.read_excel('foo.xlsx', 'Sheet1', index_col=None, na_values=['NA']))


