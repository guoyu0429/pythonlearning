#coding=utf-8
#使用k-近邻算法识别手写数字
from numpy import *
from os import listdir
import operator
import time

#准备数据     为使用数据，首先需要将32×32维的二进制数据转换为1×1024的向量，以便后续处理
def img2vector(filename):
    returnVect=zeros((1,1024))#(1,1024)元组作为zeros()函数的第一个参数，创建二维数组
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])#因为一个0或者一个1作为一个字符串，共32个，直接取lineStr[j]即可 
    return returnVect

#计算距离
def classify(inX, dataSet, labels, k):  
    #===========计算距离============  
    dataSetSize = dataSet.shape[0]       #shape函数返回array类型的各个维数上的数的个数。对2维来说，即返回行数和列数。  
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet #tile()函数如同repeat功能，dataSetSize此处为4,将输入向量inX复制4×1份  
    sqDiffMat = diffMat**2  
    sqDistances = sqDiffMat.sum(axis=1)  
    distances = sqDistances**0.5  
    sortedDistIndicies = distances.argsort()#argsort()为numpy包下的函数，返回的是数组值从小到大的索引值  
    #===========选择距离最小的k个点==  
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]#获取前K个标签
        classCount[voteIlabel]=classCount.get(voteIlabel)#先用get()函数将label I在字典里默认值为0,若无直接+1,也即label I的投票+1。
        sortedClassCount=sorted(classCount.items())
    return sortedClassCount[0][0]#返回出现次数最多标签的标签类       #排序完了之后返回一个二维列表，[0][0]代表排序最高的那个类别

#测试算法
def handwritingClassTest():
    start_time=time.time()
    hwLabels=[]
    #以往利用python os.walk()函数遍历某个文件夹下的文件，这里直接用listdir()更为方便，其返回以文件名为字符串的一个列表 
    trainingFileList=listdir('trainingDigits')
    m=len(trainingFileList)
    trainingMat=zeros((m,1024)) #同训练集一样大小的全0矩阵  
    for i in range(m):
        fileNameStr=trainingFileList[i]#某个文件的文件名，如0_0.txt 
        fileStr=fileNameStr.split(".")[0]
        classNumStr  = int(fileStr.split("_")[0])  
        hwLabels.append(classNumStr)#将某个类别存起来，以便和trainingMat对应起来，作为训练集的labels 
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)  #调用函数img2vector每行放一个1每行放一个1×1024的向量  
        
    testFileList = listdir("testDigits")  
    errorCount   = 0.0  
    mTest        = len(testFileList)  
    for i in range(mTest):  
        fileNameStr  = testFileList[i]  
        fileStr      = fileNameStr.split(".")[0]  
        classNumStr  = int(fileStr.split("_")[0])           #测试集真实类别  
        vectorUnderTest  = img2vector("testDigits/%s" % fileNameStr)  
        classifierResult = classify(vectorUnderTest, trainingMat, hwLabels, 3)  
        if i <10:  
            print ('the classifier came back with: %d, the real answer is: %d' % (classifierResult, classNumStr))  
        if classifierResult != classNumStr:  
            errorCount += 1.0  
    print ('\nthe total number of errors is: %d' % errorCount)
    print ('\nthe total error rate is: %f' % (errorCount/float(mTest))) 
    end_time = time.time()          
    print ('All time is ', end_time - start_time) 
 
print(handwritingClassTest())   