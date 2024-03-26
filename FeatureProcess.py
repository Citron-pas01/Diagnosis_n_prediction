import csv
from numpy import *
from pylab import * 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from Select import dataSet

# 因为含中文字符转成U码的utf-8时超出了其范筹，解决方法是将.decode('utf-8')改为.decode('gbk')
def readCsvFile(filepath):
    csvrows = []
    csvfile = open(filepath,'r+',encoding = 'gbk')
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        if csvreader.line_num ==2:  #读取csv文件到第三行结束
            continue
        csvrows.append(row)
    return csvrows

outputfile1 = 'E:\Ytm\Interns\Featuredp.csv'

# 读取中文的label名
label_name = readCsvFile("E:\Ytm\Interns\label_name.csv")
label_set = dataSet.columns.values.tolist() #获取合并后的数据集的列名

#---------------------------------各特征描述参数---------------------------------------------#
getMax = dataSet.loc[:,label_set].max()
getMin = dataSet.loc[:,label_set].min()
getMean = dataSet.loc[:,label_set].mean()
getStd = dataSet.loc[:,label_set].std()
def getParam(getMax):
    dp_sub = []
    dataSet_max = dataSet.apply(lambda column: getMax, axis =0) #生成的是一个对称矩阵型df,axis=为逐行取值
    for i in label_set:                 
        dp_sub.append(dataSet_max.at[i,i])   #取对角线上的值
    return pd.DataFrame(dp_sub)
# 合并各特征参数
label_sub = []
for i in label_name[0]:
    label_sub.append(i)
df_label = pd.DataFrame(label_sub)
Featuredp = pd.concat([df_label,getParam(getMax),getParam(getMin),getParam(getMean),getParam(getStd)],axis =1) 
# 加上了字符就不能够统一type，也不能进行append
Featuredp.columns=['label','Max','Min','Mean','Std']  #改数据列表列名
Featuredp.to_csv(outputfile1,encoding="utf_8_sig")
#print(Featuredp) 

#---------------------------------画相关系数云图---------------------------------------------#
correlations = dataSet.corr() 
correction=abs(correlations)# 取绝对值，只看相关程度 ，不关心正相关还是负相关
#plot correlation matrix 
fig = plt.figure(figsize =[5,5]) 
ax = fig.add_subplot(111) #图片大小为20*20
ax = sns.heatmap(correction,cmap=plt.cm.Greys, linewidths=0.05,vmax=1, vmin=0 ,annot=True,annot_kws={'size':6,'weight':'bold'})
#热力图参数设置（相关系数矩阵，颜色，每个值间隔等）
#ticks = numpy.arange(0,16,1) #生成0-16，步长为1 
mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.xticks(np.arange(20)+0.5,label_name[0],fontsize=7,rotation = 45) #横坐标标注点
plt.yticks(np.arange(20)+0.5,label_name[0],fontsize=7,rotation = 20) #纵坐标标注点
ax.set_title('Characteristic Correlation')#标题设置
plt.savefig('cluster.png',dpi=300,)
#plt.show()

#-----------------------------------标准化处理----------------------------------------------#
newDataSet = dataSet.apply(lambda x:(x -np.mean(x))/(np.std(x)))

#------------------------------------异常值处理--------------------------------------------#
def detect_outliers(df,n,features):
    outlier_indices = []  
    # 进行特征迭代
    for col in features:    
        Q1 = np.percentile(df[col], 25)  # 四分之一 (25%)
        Q3 = np.percentile(df[col],75)  # 四分之三 (75%)
        IQR = Q3 - Q1   # 四分间距 (IQR)
        outlier_step = 1.5 * IQR    # outlier step
        # 获取特征列的异常值
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        # 将获得的异常连接起来
        outlier_indices.extend(outlier_list_col)
    outlier_indices = Counter(outlier_indices)   #选出数量大于2的异常值     
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    return multiple_outliers 
Outliers_to_drop = detect_outliers(newDataSet,2,label_set)
#print(newDataSet.loc[Outliers_to_drop])
newDataSet = newDataSet.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)    #去掉异常值