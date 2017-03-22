#coding=utf-8
#k近邻算法

from numpy import *
import operator


#特征值和标签值
def createDataSet():
    group  = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]]) 
    labels = ['A','A','B','B']
    return group,labels

def classify(inX,dataSet,labels,k):#inx为待分类的输入向量，dataSet为训练样本，labels为训练样本的标签向量，k表示最近邻居的数目。  欧式距离公式
    #计算距离
    dataSetSize = dataSet.shape[0]#得到数组的行数，即知道有几个训练数据
    diffMat = tile(inX,(dataSetSize,1))-dataSet #tile将原来的一个数组，扩充成了4个一样的数组。diffMat得到了目标与训练数值之间的差值
    sqDiffMat = diffMat**2#各个元素平方
    sqDistances = sqDiffMat.sum(axis=1)#对应列相加，即得到了每一个距离的平方
    distances = sqDistances**0.5 #开方，得到距离
    sortedDistIndicies = distances.argsort()#升序排列
    print(sortedDistIndicies)
    
    #选择距离最小的K个点
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]#获取sortedDistIndicies[i]的标签
        print(voteIlabel)
        #if currentLabel not in labelCounts.keys():  
        #   labelCounts[currentLabel] = 0  
        #labelCounts[currentLabel] += 1
        classCount[voteIlabel]=classCount.get(voteIlabel)#先用get()函数将label I在字典里默认值为0,若无直接+1,也即label I的投票+1。
        sortedClassCount=sorted(classCount.items())
        print(sortedClassCount)
    return sortedClassCount[0][0]#返回出现次数最多标签的标签类
    
if __name__=="__main__":  
    group,labels = createDataSet()  
    target = classify([0,0],group,labels,3)  
    print (target)