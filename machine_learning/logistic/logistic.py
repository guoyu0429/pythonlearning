#coding=utf-8
#logistic回归
from numpy import *
import matplotlib.pyplot as plt
#logistic回归梯度上升优化算法

#打开文本文件并逐行读取
def loadDataSet():
    dataMat=[]#读取每行的前两个值X1，X2，同时将X0的值设为1
    labelMat=[]#读取类别标签
    fr=open('testSet.txt')
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

#梯度上升算法
#计算量大
def gradAscent(dataMatIn,classLabels):#第一个参数是一个100*3的矩阵，每一行代表X0，X1，X2
    dataMatrix=mat(dataMatIn)#转换成numpy矩阵
    labelMat=mat(classLabels).transpose()#类别标签：1*100的行向量
    m,n=shape(dataMatrix)#得到矩阵的大小
    alpha=0.001#向目标移动的步长
    maxCycles=500#迭代次数
    weights=ones((n,1))
    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights)
        error=(labelMat-h)
        weights=weights+alpha*dataMatrix.transpose()*error
    return weights

#随机梯度上升,计算参数
def stocGradAscent0(dataMatrix,classLabels,numIter=150):#默认迭代次数为150次
    m,n=shape(dataMatrix)
    weights=ones(n)
    for j in range(numIter):
        dataIndex=list(range(m))
        for i in range(m):
            alpha=4/(1.0+i+j)+0.01;      # 步长不是定值
            randIndex=int(random.uniform(0,len(dataIndex)))
            h=sigmoid(sum(dataMatrix[randIndex]*weights))
            error=(classLabels[randIndex]-h)
            weights=weights+alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights
   
#画出决策边界
def plotBestFit(wei):
    weights=wei#矩阵通过这个getA()这个方法可以将自身返回成一个n维数组对象
    dataMat,labelMat=loadDataSet()
    dataArr=array(dataMat)
    n=shape(dataArr)[0]
    xcord1=[]
    ycord1=[]
    xcord2=[]
    ycord2=[]
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    
    
   
#dataArr,labelMat=loadDataSet()
#weights=gradAscent(dataArr, labelMat)
#weights=stocGradAscent0(array(dataArr),labelMat)
#plotBestFit(weights)   