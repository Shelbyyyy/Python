
import pandas as p#导入目前所需要的库并给与简称
data_train = '../homework/train.csv' #查看基本数据
data_train = p.read_csv(data_train)#导入训练模型
print(data_train.info())#查看数据类型
print(data_train.describe())#粗略查看基本数据


###导入并且查看原始数据



import matplotlib.pyplot as pt
import numpy as n
pt.rcParams['font.sans-serif']=['Simhei'] #解决中文为方块的问题
pt.rcParams['axes.unicode_minus'] = False #解决图像是负号显示为方块的问题
fig = pt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

pt.subplot2grid((2,3),(0,0))  # 在一张大图里分一些小图并设定位置
data_train.Survived.value_counts().plot(kind='bar') #以生存总数为标准 设置图标种类为柱状图
pt.title("生存 (1 Survived)") 
pt.ylabel("生存人数")  
 
pt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind="bar")
pt.ylabel("总人数")
pt.title("仓位")
 
pt.subplot2grid((2,3),(0,2))
pt.scatter(data_train.Survived, data_train.Age)
pt.ylabel("年龄")                         
pt.grid(b=True, which='major', axis='y') 
pt.title("年龄 (1 Survived)")

 
pt.subplot2grid((2,3),(1,0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')   
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
pt.xlabel("年龄")
pt.ylabel("密度") 
pt.title("各等级的乘客年龄分布")

pt.legend(('头等舱', '2等舱','3等舱'),loc='best') # 设置图例
pt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
pt.title("各登船口岸上船人数")
pt.ylabel("人数")  
pt.show()
#粗略的以数据可视化的形式更直观的查看原始数据




fig = pt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()#将未生存总数0存入value并与仓位对应
print(Survived_0)
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df=p.DataFrame({'生存':Survived_1, '未生存':Survived_0})
df.plot(kind='bar', stacked=False)
pt.title("仓位与生存率是否相关")
pt.xlabel("仓位") 
pt.ylabel("总人数") 
pt.show()
#设立假设 仓位 也就是阶级 与生存率有关与否

fig = pt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数
Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df=p.DataFrame({'男性':Survived_m, '女性':Survived_f})
df.plot(kind='bar', stacked=False)
pt.title("性别与生存率是否相关")
pt.xlabel("性别") 
pt.ylabel("总人数")
pt.show()
#设立假设 性别与生存率是否相关

fig = pt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数 
Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df=p.DataFrame({'生存':Survived_1, '未幸存':Survived_0})
df.plot(kind='bar', stacked=False)
pt.title("假设登船港口与生存率是否有关")
pt.xlabel("港口") 
pt.ylabel("总人数")  
pt.show()
#假设登船港口与生存率是否有关

g = data_train.groupby(['SibSp','Survived'])
df = p.DataFrame(g.count()['PassengerId'])
print(df)
 
g = data_train.groupby(['Parch','Survived'])
df = p.DataFrame(g.count()['PassengerId'])
print(df)
#判断是否有兄弟姐妹在船上以及是否有父母子女在船上与生存率是否有关

###设立假设 进行数据分析


### 处理空值年龄
from sklearn.ensemble import RandomForestRegressor #从sklearn库中导入随机森林
 
### 使用 RandomForest 填补缺失的年龄属性
def set_missing_ages(df):#定义函数
 
    
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]#将已有并且可用特征存入age
     
    known_age = age_df[age_df.Age.notnull()].values#将年龄根据是否为空值为判断条件 分别储存为已知和未知两种值
    unknown_age = age_df[age_df.Age.isnull()].values
 
    y = known_age[:, 0]#y为我们希望求得的空值年龄
 
    x = known_age[:, 1:]#X为我们所给予的可用特征
 
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(x, y)#利用fit将x，y放入随机森林中，并设定随机森林的属性
 
    predictedAges = rfr.predict(unknown_age[:, 1::])#用随机森林中得出的结果去预测未知年龄
 
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges #将得出的年龄存入空值中
 
    return df, rfr#返回函数

 
def set_Cabin_type(df):#定义函数以将Cabin中是否有值当成条件判断分别设置成Yes以及No
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df#返回函数
 
data_train, rfr = set_missing_ages(data_train)#将预测值存入训练样本中以供使用
data_train = set_Cabin_type(data_train)#将Yes及No存入训练样本中以供使用

data_train.info()#再次查看整理过的数据

### 处理空值港口
def set_Embarked_type(df):#以填补港口空值为出发点首先进行数据转换以便使用fillna
    df.loc[ (df.Embarked=='S'), 'Embarked' ] = "1"
    df.loc[ (df.Embarked=='C'), 'Embarked' ] = "2"
    df.loc[ (df.Embarked=='Q'), 'Embarked' ] = "3"
    return df

data_train = set_Embarked_type(data_train)

data_train.Embarked = data_train.Embarked.fillna(0)

data_train.Embarked = list(map(int,data_train.Embarked))

print(data_train.Embarked.mean())

def set_Embarked_type(df):#再将已经填补完空值的列表赋值回训练样本#
	df.loc[ (df.Embarked==0), 'Embarked' ] = "S"
	return df
data_train = set_Embarked_type(data_train)

### 使用随机森林处理票价为0的值

def set_missing_fare(df):#定义函数
 
    
    fare_df = df[['Fare','Age','Parch', 'SibSp', 'Pclass']]#将已有并且可用特征存入age
     
    known_fare = fare_df.loc[fare_df.Fare != 0].values#将年龄根据是否为0为判断条件 分别储存为已知和未知两种值
    unknown_fare = fare_df.loc[fare_df.Fare == 0].values
 
    y1 = known_fare[:, 0]#y为我们希望求得的0票价
 
    x1 = known_fare[:, 1:]#X为我们所给予的可用特征
 
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(x1, y1)#利用fit将x，y放入随机森林中，并设定随机森林的属性
 
    predictedAges = rfr.predict(unknown_fare[:, 1::])#用随机森林中得出的结果去预测未知年龄
 
    df.loc[ df.Fare == 0, 'Fare' ] = predictedAges #将得出的年龄存入空值中
 
    return df, rfr#返回函数

data_train, rfr = set_missing_fare(data_train)
print(data_train.Fare.describe())


###数据处理


### 使用算法开始建模 这里使用逻辑回归
data_train.Pclass = data_train.Pclass.astype('object')
cate =p.get_dummies(data_train[['Cabin','Sex','Embarked','Pclass']])
data_new = data_train[['Survived','Age','SibSp','Parch','Fare']].join(cate) #数据的转储以及整理
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_new.iloc[:,1:], data_new.Survived, test_size = 0.2, random_state=34)
lr = LogisticRegression()
lr.fit(x_train,y_train)#用数据X，y来训练模型
pred = lr.predict(x_test)
from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_test,pred))#预测准确率
print(accuracy_score(y_test,pred))#分类准确率分数

#尝试使用不同算法 这里使用决策树
from sklearn.tree import *
dt = DecisionTreeClassifier(random_state=99,splitter='best', presort=True)
dt.fit(x_train,y_train)
pred = dt.predict(x_test)
from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_test,pred))
print(accuracy_score(y_test,pred))

####模型构建


data_test = p.read_csv('../homework/test.csv')#导入测试样本

def set_missing_ages(df,rfr):
 
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
 
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values
 
    y3 = known_age[:, 0]#目标年龄
 
    X3 = known_age[:, 1:]#特征属性值

    predictedAges = rfr.predict(unknown_age[:, 1::])    # 用得到的模型进行未知年龄结果预测
 
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges     # 用得到的预测结果填补原缺失数据
 
    return df

data_test = set_missing_ages(data_test, rfr)
data_test = set_Cabin_type(data_test)
data_test.Pclass = data_test.Pclass.astype('object')
cate_test =p.get_dummies(data_test[['Cabin','Sex','Embarked','Pclass']])
data_test_new = data_test[['PassengerId','Age','SibSp','Parch','Fare']].join(cate_test)


final = dt.predict(data_test_new.fillna(0))
final_1=data_test[['PassengerId','Age']]
final_1['Survived'] = final
final = final_1[['PassengerId','Survived']]
final.to_csv('C:/Users/yang/Desktop/code/python/homework/6.csv')

print(final.describe())
print(data_test_new)

### 使用训练好的模型进行预测



