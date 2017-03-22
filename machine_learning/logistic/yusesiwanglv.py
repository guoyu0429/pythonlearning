#coding=utf-8
#从疝气病预测病马的死亡率
#1.数据预处理
'''
1.特征缺失的数据：所有的缺失值必须用一个实数值来替换，这里我们用0来替换（在更新时不会影响系数的值）
2.类别标签缺失的数据：采用logistic回归的话将数据丢弃即可
'''
from test1.logistic.logistic import sigmoid, stocGradAscent0
from numpy import *

#将回归系数和特征向量作为输入计算sigmoid的值
def classifyVector(inX,weights):
    prob=sigmoid(sum(inX*weights))
    if prob>0.5:
        return 1.0
    else:
        return 0.0
 
#打开测试集和训练集   ，并对数据进行格式化处理
def colicTest():
    frTrain=open('horseColicTraining.txt')
    frTest=open('horseColicTest.txt')
    trainingSet=[]
    trainingLabels=[]
    for line in frTrain.readlines():#读取每一行数据
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):#读取每行数据的21个特征
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    #计算回归系数向量
    trainWeights=stocGradAscent0(array(trainingSet),trainingLabels,500)
   
    errorCount=0
    numTestVec=0.0
    for line in frTest.readlines():
        numTestVec+=1.0
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!=int(currLine[21]):
            errorCount+=1
    errorRate=(float(errorCount/numTestVec))
    print('the error rate of this test is:%f'%errorRate)
    return errorRate

#调用函数colicTest()10次并求均值
def multiTest():
    numTests=10;
    errorSum=0.0
    for k in range(numTests):
        errorSum+=colicTest()
    print('after %d iterations the average error rate is %f'%(numTests,errorSum/float(numTests)))
        
   
multiTest()     