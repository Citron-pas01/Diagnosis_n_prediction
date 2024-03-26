import math
import time
import random
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from Distribution import newDataSet,label_name

# 指定默认字体 解决画图中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

outputfile2 = 'E:\Ytm\Interns\data_clustering.csv' # 保存分类结果的文件名

array = newDataSet.values
x = array[:, :-1]
y = array[:, -1]

#split the data
x_train, x_test,y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state= 7)
#scaler = StandardScaler()
#scaler.fit(x_train)
#x_train = scaler.transform(x_train)
#x_test = scaler.transform(x_test)

#refer report section under "PCA"
#make instance of the model
pca = PCA(.95)
pca.fit(x_train)

#to see how many components were selected
total_no_comp = pca.n_components_
print(total_no_comp)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)

neighbors = np.arange(1,10)
train_acc = np.empty(len(neighbors))
test_acc = np.empty(len(neighbors))

#run code to get accuracies per k
for i, k in enumerate(neighbors):
  knn= KNeighborsClassifier(n_neighbors = k)
  knn.fit(x_train, y_train.astype(int))
  train_acc[i] = knn.score(x_train, y_train.astype(int))
  test_acc[i] = knn.score(x_test, y_test.astype(int))

#plot the values
plt.plot(neighbors, train_acc, label= "train set accuracy")
plt.plot(neighbors, test_acc, label= "test set accuracy")
plt.legend()
plt.xlabel("No. of neighbors")
plt.ylabel("Accuracy")
plt.title("kNN accuracy with varying number of neighbors")
plt.show()

# 模型训练 KNN
k = 6
knn = KNeighborsClassifier(n_neighbors = k )
knn.fit(x_train, y_train.astype(int))
y_pred= knn.predict(x_test)
print(knn.score(x_test, y_test.astype(int)))
print(metrics.confusion_matrix(y_test.astype(int), y_pred))
print(metrics.classification_report(y_test.astype(int), y_pred))

#-----------------------------------------用Kmeans聚类进行评估--------------------------------------#
from sklearn.cluster import KMeans
clusters = np.arange(1,10)
train_acc = np.empty(len(clusters))
test_acc = np.empty(len(clusters))

for i, k in enumerate(clusters):
  ClusteringModel = KMeans(n_clusters = k, n_jobs = 4, max_iter = 500) #分为k类，并发数4
  ClusteringModel.fit(x_train, y_train.astype(int))
  train_acc[i] = ClusteringModel.score(x_train, y_train.astype(int))
  test_acc[i] = ClusteringModel.score(x_test, y_test.astype(int))

plt.plot(clusters, train_acc, label= "train set accuracy")
plt.plot(clusters, test_acc, label= "test set accuracy")
plt.legend()
plt.xlabel("No. of clusters")
plt.ylabel("Accuracy")
plt.title("kmeans accuracy with varying number of neighbors")
plt.show()

#-------------------------------------原始数据可视化分析---------------------------------------#

fig = plt.figure(figsize=(8,8))
plt.suptitle("Dimensionality Reduction and Visualization",fontsize =14)
#t-sne的降维与可视化
tsne = manifold.TSNE(init='pca',perplexity=30,random_state=0)
#训练模型
#for i in range(0,3):
z = tsne.fit_transform(newDataSet)
ax1 = fig.add_subplot()
plt.scatter(z[:,0],z[:,1],cmap=plt.cm.Spectral)
ax1.set_title('Pristine Plot',fontsize=14)
plt.show()

#------------------------------------族心数为6的聚类情况--------------------------------------#
ClusteringModel = KMeans(n_clusters = 7, n_jobs = 4, max_iter = 500) #分为k类，并发数4
ClusteringModel.fit(newDataSet) #开始聚类
r1 = pd.Series(ClusteringModel.labels_).value_counts() #统计各个类别的数目
r2 = pd.DataFrame(ClusteringModel.cluster_centers_) #找出聚类中心
r = pd.concat([r2, r1], axis = 1) #横向连接（axis=0是纵向），得到聚类中心对应的类别下的数目
r.columns = list(newDataSet.columns) + [u'NumClustering'] #重命名表头
print(r)

r = pd.concat([newDataSet, pd.Series(ClusteringModel.labels_, index = newDataSet.index)], axis = 1)  #详细输出每个样本对应的类别
r.columns = list(newDataSet.columns) + [u'NameClustering'] #重命名表头
r.to_csv(outputfile2)

tsne.fit_transform(newDataSet) #进行数据降维       
tsne = pd.DataFrame(tsne.embedding_, index = newDataSet.index)

d = tsne[r[u'NameClustering'] == 0]
ax1 = fig.add_subplot(2,1,2)
plt.scatter(d[0],d[1],cmap=plt.cm.Spectral)
ax1.set_title('Clustering Plot',fontsize=14)

d = tsne[r[u'NameClustering'] == 1]
ax1 = fig.add_subplot(2,1,2)
plt.scatter(d[0],d[1],cmap=plt.cm.Spectral)
ax1.set_title('Clustering Plot',fontsize=14)

d = tsne[r[u'NameClustering'] == 2]
ax1 = fig.add_subplot(2,1,2)
plt.scatter(d[0],d[1],cmap=plt.cm.Spectral)
ax1.set_title('Clustering Plot',fontsize=14)

d = tsne[r[u'NameClustering'] == 3]
ax1 = fig.add_subplot(2,1,2)
plt.scatter(d[0],d[1],cmap=plt.cm.Spectral)
ax1.set_title('Clustering Plot',fontsize=14)

d = tsne[r[u'NameClustering'] == 4]
ax1 = fig.add_subplot(2,1,2)
plt.scatter(d[0],d[1],cmap=plt.cm.Spectral)
ax1.set_title('Clustering Plot',fontsize=14)

d = tsne[r[u'NameClustering'] == 5]
ax1 = fig.add_subplot(2,1,2)
plt.scatter(d[0],d[1],cmap=plt.cm.Spectral)
ax1.set_title('Clustering Plot',fontsize=14)

d = tsne[r[u'NameClustering'] == 6]
ax1 = fig.add_subplot(2,1,2)
plt.scatter(d[0],d[1],cmap=plt.cm.Spectral)
ax1.set_title('Clustering Plot',fontsize=14)
plt.show()
