from numpy import *
import pandas as pd
import numpy as np

data_qz = pd.read_csv("E:\Ytm\Interns\pingshan1.csv")
data_index = pd.read_csv("E:\Ytm\Interns\select.csv")#选取测点集数据的label文档
df = pd.DataFrame(data_qz)
df_index = np.array(data_index)
label =np.array(df.columns.values.tolist())
#分配单个测点的数据
d1 = df_index[:,0]
d1 = filter(lambda v: v==v,d1) #删去array中的NAN值
singleDectorSet = df.loc[:,d1]
#setOutPortValue('out', singleDectorSet)

#----------------------------------分配多重测点（1，2，3，4，5）---------------------------------#
def Distribution(n):
    dp_set = df_index[:,n]
    dp_set = filter(lambda v: v==v,dp_set)
    detector = df.loc[:,dp_set]
    corref = detector.corr()
    numCorref = corref.values
    if numCorref.min()<0.9:
        print("同属多重测点数据相关性较差，需进一步分析并通过权重降维！")
    else:   #有点冗杂，需进一步精简
        for (i,j) in range(len(df_index[:,n]),len(df_index[:,n])):
            detector = (detector[i]+ detector[i])/(2*len(df_index[:,n]))
    return detector.loc[:,df_index[1,n]] 
#print(Distribution(1))
#将各分类去重的测点数据进行拼接
dataSet = pd.concat([singleDectorSet,Distribution(1),Distribution(2),Distribution(3),Distribution(4),Distribution(5)],axis=1)
#print(dataSet)

