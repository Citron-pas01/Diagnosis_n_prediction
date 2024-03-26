#import relevant libraries
import pandas
import numpy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from FeatureProcess import newDataSet,label_name
import matplotlib.pyplot as plt

# 将 y 值调到最后一列
newDataSet.head()
newDataSet_out = newDataSet['20MAA50CT025A']
newDataSet.drop(labels ='20MAA50CT025A', axis = 1, inplace = True)
newDataSet.insert(newDataSet.shape[1],'20MAA50CT025A', newDataSet_out)
#print(newDataSet)

#--------------------------------分解成训练集和测试集-------------------------------------#
X = newDataSet.iloc[:, :-1].values   # 因变量选择
y = newDataSet.iloc[:, -1].values     # 函数值
features = newDataSet.drop(["20MAA50CT025A"], axis=1) # 换成平衡活塞温度去掉
validation_size = 0.4
seed = 7
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=validation_size, random_state=seed)

'''
#决策树预测分类
dtree = DecisionTreeClassifier(criterion="entropy", random_state=123,max_depth=4,min_samples_leaf=5)
dtree.fit(X_train, y_train.astype(int))
pred_train = dtree.predict(X_train)
pred_test = dtree.predict(X_test)'''


###------------随机森林------------###
#训练模型
forest = RandomForestClassifier(n_estimators=10, criterion="entropy",max_depth=4, min_samples_leaf=5)
forest.fit(X_train, y_train.astype(int))
 
# 预测
pred_train = forest.predict(X_train)
pred_test = forest.predict(X_test)
###----------------------------------


#准确率分析(可共用）
train_acc = accuracy_score(y_train.astype(int), pred_train)
test_acc = accuracy_score(y_test.astype(int), pred_test)

'''
print ("训练集准确率: {0:.2f}, 测试集准确率: {1:.2f}".format(train_acc, test_acc))
#其他模型评估指标
precision, recall, F1, _ = precision_recall_fscore_support(y_test.astype(int), pred_test, average="micro")
print ("精准率: {0:.2f}. 召回率: {1:.2f}, F1分数: {2:.2f}".format(precision, recall, F1))
#特征重要度
importances = dtree.feature_importances_
indices = np.argsort(importances)[::-1]
num_features = len(importances)'''


###-------------随机森林--------------###
#其他模型评估指标
precision, recall, F1, _ = precision_recall_fscore_support(np.array(y_test.astype(int)), np.array(pred_test), average='micro')
#特征重要度
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
num_features = len(importances)
###----------------------------------###


label_features = label_name[0]
label_features.remove("平衡活塞后蒸汽温度 #1") 
#将特征重要度以柱状图展示
plt.figure()
plt.title("Feature importances")
plt.bar(range(num_features), importances[indices], color="g", align="center")
plt.xticks(range(num_features), [label_features[i] for i in indices], rotation='45')
plt.xlim([-1, num_features])
plt.show()
 
#输出各个特征的重要度
#for i in indices:
    #print ("{0} - {1:.3f}".format(features[i], importances[i]))

#end of code
