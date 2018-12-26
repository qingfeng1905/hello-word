""" Python学习笔记 """
#1. numpy和pandas是用c语言写的，pandas基于numpy;
#2. 文件头部指定编码方式，UTF-8是Unicode的一种实现方式;
#3. Python中默认的编码格式是ASCII格式，在没修改编码格式时无法正确打印汉字。

# Anaconda：常用命令
!conda info #系统信息
!conda list #查看所有安装包
!pip list #查看所有安装的包

# 例子：安装包
!pip install tushare #安装包
import tushare as ts
# 升级
!pip -m pip install --upgrade pip

# 万得接口，wind
import WindPy as api
api.w.start()
api.w.isconnected()

# 设置工作目录路径
import os
os.getcwd() #获取当前工作目录
os.chdir('G:\\xx\\yy') #设置工作目录 

cd G:\\OnePiece\\Quant
help(list)	#list函数的帮助

ls	#查看当前目录下文件列表
os.listdir() #返回当前目录下的文件列表

who	#查看工作区变量
whos 	#查看工作区变量详细信息
dir()	#whos的加强版，包含系统变量

%clear	#清空命令窗口，快捷键 Ctrl + L
reset	#清空所有变量
del var1, var2	#删除特定变量

#从命令窗口输入参数
x = int(input('pls input a number：'))

id(var)	 #查看变量在内存中的位置
type(var) 	#查看数据类型
isinstance(var, str) 	#判断变量类型


"""流程控制语句
break  #跳出循环
continue #结束本次循环，进入下一次
pass #代码桩，占位用的，保持结构完整
return #
"""

c = 2 + 3.15j 	#复数类型

#字符串，str
x = "hello"
for i in x: print(i)

#元组：tuple，不可更改
y = ('a', 'b', 3.14)

#列表：list；note：两个列表不能直接相乘
z = ['a', 'b', 3.14]


#字典：dict
d = {'name':'milo', 'age':28}
d['age']
d.keys()
d.values()
#用两个列表生成字典
xx = dict(zip(listA,listB)) # 持仓债券：债务 

#range
range(start, end, step)


#函数定义&调用
def fun(x1, x2, *args1, **args2):
	print("define a function")
	return x1*x2

fun(x1,x2)

#匿名函数
sum = lambda x1,x2: x1+x2 #定义
sum(x1,x2)	#调用

#if语句
if 1>2:
	print('ok')
elif 2<3:
	print('hello world')
else:
	print('fuck the world')
	
#for语句
for x in range(0,10):
	pring(x**2)
else:
print('end')

#while语句
x = 2
while x<100:
	print(x**2)
	x = x**2
else:
	print('end')


#正则表达式
import re
xx = r"t[abc]"
re.findall(xx,"whatafuckingday")

#字符串中提取数字
import re   
ss = '123ab45'
xx = re.sub("\D","",ss)

#字符创剔除字符
xx['证券代码'].str.lstrip('yh') + '.IB'


#变量长度
len(df)


# 填充，补齐位数
tmp = str(data.loc[idx,'证券代码'])
tmp = tmp.zfill(6)


#文件读写相关函数
open()
read()
write()
close()
readline()
readlines()
writeline()
writelines()
seek()
seek(0,0)
flush()





""" //////////////////////////////////////////////////////////////////
numpy
//////////////////////////////////////////////////////////////////"""
import numpy as np

#创建一些基础矩阵
m = np.array([[1,2,3],[4,5,6]])

x = [[1,2,3],[4,5,6]]
x = ([1,2,3],[4,5,6])
m = np.array(x) #用tupple和list创建数组

#特殊矩阵创建
m = np.zeros((9,9))
m = np.ones((8,8))
m = np.eye(3)

#其他矩阵创建
m = np.arange(1,10,2)
m = np.arange(9)
m = np.arange(10).reshape((2,5))
m = np.linspace(0,100,5) #5个彼此间隔相等的点

#矩阵属性
m.ndim 	#矩阵是几维的
m.shape #返回行和列数
m.size 	#元素个数
m.dtype 	#元素类型

np.pi
np.e

#常用函数
np.sin()
np.sqrt()

x = np.random.random((2,4))
x.sum()
x.min()
x.max()

#矩阵乘法
x = np.ones((2,2))
y = np.array(([1,3],[2,4]))
z = np.dot(x,y) #乘法
z = x.dot(y)

#矩阵合并
z = np.vstack((x,y)) #纵向合并
z = np.row_stack((x,y))

z = np.hstack((x,y)) #横向合并
z = np.column_stack((x,y))

#nan
np.isnan(x) #判断是否nan
np.isnan(df.loc[9,'y']) #是否nan

#筛选，定位
row,col = np.where(x==0) #返回行列位置

#去除重复值
code_ = list(df_bond['windCode'])
code = list( np.unique(code_) )

# 用集合去除重复值
trdDate = list( set(df['日期']) ) 

#其他函数
np.split
np.array_split
np.vsplit
np.hsplit
np.concatenate
np.sort
np.nonzero
np.transpose
np.clip



""" //////////////////////////////////////////////////////////////////
pandas

#十分钟搞定pandas
http://www.cnblogs.com/chaosimple/p/4153083.html
?
#时间序列处理
http://blog.csdn.net/qq_36330643/article/details/78335399
//////////////////////////////////////////////////////////////////"""
import pandas as pd

#创建DataFrame
dates = pd.date_range(start='20180128', periods=30)
df = pd.DataFrame(np.arange(60).reshape(30,2), index=dates, columns=['a','b'])
df.index
df.columns
df.values
df.describe() #基本统计量

isinstance(df.loc[9,'a'], str) #判断类型
type(df.loc[9,'a'])

#首行，最后一行
df.iloc[0,]
df.iloc[-1,]

#基本的一些使用
y = pd.Series(range(1,10))
y.tolist()

df['new_col'] = np.nan #新增一列
df.loc[:,'new_col'] = ['aa','bb','cc','dd','ee'] #新增一列

#选择一列
df['a']
df.a
#选择行：注意下面两种方法的区别，一个结果是一行，一个是两行
#
df[0:1] #1行
df['20170127':'20170128'] #2行

df[0:2]['a'] #指定行、列

#select by label: loc
df.loc['20170127']
df.loc[:,'a']

#select by position: iloc
df.iloc[3,0]
df.iloc[0:2,0:1] #两行、一列

#mixed selection: ix
"""
在ix中，分为两种模式，一种是有index，一种没有index(默认整数)
会首先判断和index的格式是否匹配，如果匹配，则是两边都包括；
如果不匹配，就是作为位置索引出现，最后一位不包括
"""
df.ix[0:3, 'b']


#pandas自带的画图
df = pd.DataFrame(np.random.randn(1000,4), index = np.arange(1000), columns = list("ABCD"))
df = df.cumsum()
df.plot()


#列重命名
df.columns = ["a", "b", "c", "d"] #直接修改
df.rename( columns={'a':'A', 'b':'BB'}, inplace=True )

#数据检查，是否存在nan
df.isnull().values.any() #返回false，无nan数据；反之，则存在
df.isnull().values.sum() #nan的数量
df.isnull().sum()

#数据筛选&删除
df = data[data['y'].isnull().values==False] #删除nan数据
df[df['y'].isnull().values==True] #返回nan值的位置

xx = df[ (df['x1']==0) & (df['x2']==0)]

#集合，筛选
df['a'].isin(alist)

#增加一列
data['yy'] = data.index #会出现警告，可采用下面插入一列的方式，还可以制定插入位置
data.insert(1,'yy', data.index) 

#删除一列
df = df.drop(['xx'], axis=1, inplace=True) #会出现警告，可采用下面的方式
df = df.drop(['xx'], axis=1)
del df['xx']

#缺失值处理，nan
df.dropna(axis=0) #删除带有nan的行
df.fillna(0) #用0填充nan
df.fillna('stt') #用字符串填充
df.fillna(method='pad') #用前一个数据填充
df.fillna(method='bfill') #用后一个数据填充
df.fillna(df.mean()) #使用一个统计量来代替
df.fillna(df.mean()['one','two']) #指定特定的列进行缺失值处理

#合并
"""
http://pandas.pydata.org/pandas-docs/stable/merging.html
"""
pd.concat([df1,df2], axis=1) #横向拼接:
result = pd.concat([df1, df2], axis=1, join_axes=[df1.index])
pd.append()
pd.merge()
pd.join()

xx = pd.merge(finance_all, df, how='left', on=['code'])
yy = pd.merge(xx, sample, how='left', on=['发行主体','rptDate'])

#日期批量处理：从字符串格式的日期序列中，提取年、月、日
datetime_index = pd.to_datetime(df['date'], format='%Y%m%d')
df.index = datetime_index
year = df.index.year
month = df.index.month
day = df.index.day

#日期批量处理：日趋转化为指定格式的字符串
datestr = df.index.strftime('%Y-%m-%d %H:%M:%S')

#日期批量处理
dates = pd.date_range('20170101', periods = 20)
year = list(dates.year)
xx = dates.strftime('%Y-%m-%d %H:%M:%S')

#日期批量处理
listedStock['ipo_date'] = listedStock['ipo_date'].dt.strftime('%Y%m%d')

#删除特定行
df.drop(i, inplace=True) #删除第i行

tmp = df[ (df['x1']==0) & (df['x2']==0)]
df.drop(tmp.index, inplace=True)

#删除特定列
del df['column_name']
df.drop('column_name', axis=1, inplace=True)
df.drop(df.columns[i], axis=1, inplace=True)

#排序
df.sort_values(by=['column_name'], ascending=False, inplace=True)

#时间序列：滞后n阶
df['a'].shift(n)

#数据聚合，groupby, 分类汇总
group = df.groupby(['industry_sw'])
xx = group['a'].count()
yy = group['a'].agg(['count','sum','min','max']) #同时作用多个函数

#读取Excel
df = pd.read_excel('data.xlsx', sheetname='Sheet1', index_col=0)
df = pd.read_excel('data.xlsx', sheetname='Sheet1', index_col=0, header=None)

#存Excel: 如果不使用writer，则会覆盖原来的表单，只有最新的
writer = pd.ExcelWriter('data.xlsx')
df.to_excel(writer, sheet_name='Sheet1', index=False)
df.to_excel(writer, sheet_name='abc', index=False)
writer.save()

#读取csv
df = pd.read_csv('data.csv', header=None, index_col=0, encoding='gbk')

#存csv
df.to_csv('test.csv',index=False, header=None)


""" //////////////////////////////////////////////////////////////////
画图
ref: Python--matplotlib绘图可视化知识点整理
https://www.cnblogs.com/zhizhan/p/5615947.html> 
//////////////////////////////////////////////////////////////////"""
import matplotlib.pyplot as plt
#中文字符支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 单幅图
plt.figure(1)
plt.plot(range(1,10), 'o')
plt.title('test plot')
plt.xlabel('time')
plt.ylabel('number')
plt.legend('a')
?
#可以继续叠加画图，但对单个曲线不好控制
plt.plot(np.random.random(9)) 
plt.legend('norm')

#单幅图，更易于掌控的一种方法
fig = plt.figure(1)
fig.suptitle('main_title')
#子图
ax1 = fig.add_subplot(2,2,1) #(1,1,1)则为一张图
ax1.plot(np.random.random(20), 'r-', label = 'yy')
ax1.legend() #调用才会显示legend,可指定显示位置
# ax1.legend(['yy']) #添加图例的另一种方法
ax1.clear() #清除子图
#子图标题，label
plt.title('test')
plt.xlabel('time')
plt.ylabel('number')

#subplot
plt.figure(1)                # 第一张图
plt.subplot(211)             # 第一张图中的第一张子图
plt.plot([1,2,3])
?
plt.subplot(212)             # 第一张图中的第二张子图
plt.plot([4,5,6])
?
plt.figure(2)                # 第二张图
plt.plot([4,5,6])            # 默认创建子图subplot(111)
?
plt.figure(1)                # 切换到figure 1 ; 子图subplot(212)仍旧是当前图
plt.subplot(211)             # 令子图subplot(211)成为figure1的当前图
plt.title('Easy as 1,2,3')   # 添加subplot 211 的标题
?
#双纵坐标画图         
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(np.random.random(10))
?
ax2 = ax1.twinx() #关键，创建次坐标轴
ax2.plot(range(10), 'r')
?
# 调整边界：borders
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)
?
# 其他画图函数
plt.show
plt.scatter
plt.bar
plt.hist
plt.kde
plt.area

#其他例子
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(x, y, 'o', label="data")
ax.plot(x, result.fittedvalues, 'r--.', label="OLS")
ax.legend(loc='best')



""" //////////////////////////////////////////////////////////////////
时间模块
//////////////////////////////////////////////////////////////////"""
import time
time.time()
time.localtime()
time.sleep(10)
?
import datetime
datetime.datetime.today()


""" //////////////////////////////////////////////////////////////////
数理统计，回归模型
//////////////////////////////////////////////////////////////////"""
from scipy import stats

#正态分布：累积分布函数
stats.norm(0,1).cdf(2) - stats.norm(0,1).cdf(-1)	#2倍sigma，95.45%
stats.norm(0,1).cdf(1.5) - stats.norm(0,1).cdf(-1.5) #1.5倍sigma，86.64%
stats.norm(0,1).cdf(1) - stats.norm(0,1).cdf(-1)	#1倍sigma，68.27%

#正态分布：分位数函数
icdf = stats.norm(0,1).ppf(0.5)
#样本数据分位数
icdf = np.percentile(yield_series, 5) # 5%

#生成随机数
np.random.seed(123456) #要每次产生随机数相同就要设置种子
np.random.rand(10000) #生成1万个[0,1]之间的随机数
e = np.random.normal(size=1000) #正态分布随机数

#线性回归
#from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
y = df['yy']   #因变量
x = df[['x3']]   #自变量             
x = sm.add_constant(x) #增加常数项
model = sm.OLS(y,x).fit()               
model.summary()
model.params
y_fitted = model.fittedvalues

#带有哑变量的回归，回归中的行业变量：dummy variable
#生成m个虚拟变量后，只要引入m-1个虚拟变量到数据集中，未引入的一个是作为基准对比的，如何体现的？
nsample = 50
groups = np.zeros(nsample,int)
groups[20:40] = 1
groups[40:] = 2
dummy = sm.categorical(groups, drop=True)
x = np.linspace(0,20,nsample)
X = np.column_stack((x,dummy))
X = sm.add_constant(X)
beta = [10, 1, 1, 3, 8]
e = np.random.normal(size=nsample)
y = np.dot(X, beta) + e
result = sm.OLS(y,X).fit()
print(result.summary())
#
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(x, y, 'o', label="data")
ax.plot(x, result.fittedvalues, 'r--.', label="OLS")
ax.legend(loc='best')



""" //////////////////////////////////////////////////////////////////
万得数据接口
//////////////////////////////////////////////////////////////////"""
import WindPy as api
api.w.start()
api.w.isconnected()
#功能函数
data = api.w.wsd() #日期序列
data = api.w.wss() #多维序列
data = api.w.wsi() #分钟序列
data = api.w.wsq() #实时行情
#数据格式转换：ex...
df = pd.DataFrame(data.Data)
df.index = data.Times
df.columns = data.Codes


""" //////////////////////////////////////////////////////////////////
@brief: 函数模块加载
1.默认情况下，模块在第一次被导入之后，其他的导入都不再有效;
2.如果此时在另一个窗口中改变并保存了模块的源代码文件，也无法更新该模块;
3.因为：导入是一个开销很大的操作（找到文件，编译成字节码，并运行代码）；
4.在重载之前,请确保已经成功地导入了这个模块。
//////////////////////////////////////////////////////////////////"""
from imp import reload
import fx
reload(fx)

