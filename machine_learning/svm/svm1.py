#coding=utf-8
#基于SVM的手写数字识别
from os import listdir
from numpy import *
from test1.svm.svm import smoP, kernelTrans

def loadImages(dirName):
    hwLabels = []
    trainingFileList = listdir(dirName)           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)
        else: hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels    

def testDigits(kTup=('rbf',10)):
    dataArr,labelArr=loadImages('trainingDigits')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat=mat(dataArr);labelMat=mat(labelArr).transpose()
    #取得支持向量的索引
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd]
    labelSV=labelMat[svInd]
    print('there are %d support vectors' %shape(sVs)[0])
    m,n=shape(datMat)
    errorCount=0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        #计算输出公式
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]):
             errorCount+=1.0
    print('the training error rate is:%f' %(float(errorCount)/m))
    dataArr,labelArr=loadImages('testDigits')
    errorCount=0
    for i in range(m):
        kernelEval=kernelTrans(sVs,datMat[i,:],kTup);
        predict=kernelEval.T*multiply(labelSV,alphas[svInd])+b;
        if sign(predict)!=sign(labelArr[i]):
            errorCount+=1.0;
    print('the test error rate is:%f '%(float(errorCount)/m));     
        
def img2vector(filename):
    returnVect=zeros((1,1024))#(1,1024)元组作为zeros()函数的第一个参数，创建二维数组
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])#因为一个0或者一个1作为一个字符串，共32个，直接取lineStr[j]即可 
    return returnVect

testDigits(('rbf',10))