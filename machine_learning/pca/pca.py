#coding=utf-8
'''
利用CPA主成分分析来简化数据
去除平均值  计算协方差矩阵   计算协方差矩阵的特征值和特征向量
将特征值从大到小排序
保留最上面的N个特征向量
将数据转换到上述N个特征向量构建的新空间中
'''
from numpy import *

#1.导入数据
def loadDataSet(fileName,delim='\t'):
    fr=open(fileName)
    stringArr=[line.strip().split(delim) for line in fr.readlines()]
    datArr=[list(map(float,line)) for line in stringArr]
    return mat(datArr)
#2.主成分分析
def pca(dataMat,topNfeat=9999999):
    #计算原始数据集平均值
    meanVals=mean(dataMat,axis=0)
    #减去平均值
    meanRemoved=dataMat-meanVals
    #计算协方差矩阵
    covMat=cov(meanRemoved,rowvar=0)
    #计算特征值
    eigVals,eigVects=linalg.eig(mat(covMat))
    #对特征值从小到大排序
    eigValInd=argsort(eigVals)
    #排序结果逆序，得到topNfeat个特征
    eigValInd=eigValInd[:-(topNfeat+1):-1]
    redEigVects=eigVects[:,eigValInd]
    #将数据转换到新空间
    lowDDataMat=meanRemoved*redEigVects
    reconMat=(lowDDataMat*redEigVects.T)+meanVals
    return lowDDataMat,reconMat

#3.示例：利用PCA对半导体制造数据降维
#其中数据缺失的部分将Nan替换成平均值的函数
def replaceNanWithMean():
    datMat=loadDataSet('secom.data')
    #计算特征数目
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        #对于每个特征先计算那些非NAN值的平均值
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i])
        #将所有NAN替换成该平均值
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal
    return datMat

'''
dataMat=loadDataSet('testSet.txt')
lowDMat,reconMat=pca(dataMat,1)
shape(lowDMat)
import matplotlib
import matplotlib.pyplot as plt
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker='^',s=90)
ax.scatter(reconMat[:,0].flatten().A[0],reconMat[:,1].flatten().A[0],marker='o',s=50,c='red')
plt.show()
'''
dataMat=replaceNanWithMean()
meanVals=mean(dataMat,axis=0)
meanRemoved=dataMat-meanVals
covMat=cov(meanRemoved,rowvar=0)
eigVals,eigVects=linalg.eig(mat(covMat))
print(eigVals)
