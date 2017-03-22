#coding=utf-8
#AdaBoost分类器:通过组合多个分类器进行分类
from numpy import *
import matplotlib.pyplot as plt

def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

def loadDataSet(fileName):
    numFeat=len(open(fileName).readline().split('\t'))
    dataMat=[]
    labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#用于测试是否有某个值小于或者大于我们正在测试的值,通过阈值比较对数据进行分类
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#just classify the data
    retArray = ones((shape(dataMatrix)[0],1))#将返回数组的全部元素设置为1
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

#在一个加权数据集中循环，并找到具有最低错误率的单层决策树
def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0;#用于在特征的所有可能值上进行遍历
    bestStump = {};#存储给定权重向量D时所得到的最佳单层决策树
    bestClasEst = mat(zeros((m,1)))
    minError = inf #init error sum, to +infinity
    for i in range(n):#遍历所有特征
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
        stepSize = (rangeMax-rangeMin)/numSteps#通过最小值和最大值来计算步长
        for j in range(-1,int(numSteps)+1):#loop over all range in current dimension
            for inequal in ['lt', 'gt']: #go over less than and greater than
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#call stump classify with i, j, lessThan
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr  #calc total error multiplied by D
                #print ('split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f' % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst#返回具有最小错误率的单层决策树，最小错误率和估计的类别向量

#基于单层决策树的AdaBoost的训练过程
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr=[]
    m=shape(dataArr)[0]
    D=mat(ones((m,1))/m)#存储每个数据点的权重，一开始所有都初始或为1/m
    aggClassEst=mat(zeros((m,1)))#记录每个数据点的类别估计累计值
    for i in range(numIt):#numIt代表迭代次数
        bestStump,error,classEst=buildStump(dataArr, classLabels, D)#建立一个单层决策树
        print('D:',D.T)
        alpha=float(0.5*log((1.0-error)/max(error,1e-16)))#告诉总分类器本次单层决策树输出结果的权重，max(error,1e-16)用于确保在没有错误时不会发生除零溢出
        bestStump['alpha']=alpha#该字典包括了分类所需要的所有信息
        weakClassArr.append(bestStump)
        print('classEst:',classEst.T)
        #为下一次迭代计算D
        expon=multiply(-1*alpha*mat(classLabels).T,classEst)
        D=multiply(D,exp(expon))
        D=D/D.sum()
        #错误率累加计算
        aggClassEst+=alpha*classEst
        print('aggClassEst:',aggClassEst.T)
        aggErrors=multiply(sign(aggClassEst)!=mat(classLabels).T,ones((m,1)))
        errorRate=aggClassEst.sum()/m
        print('total error:',errorRate,'\n')
        if errorRate==0.0:break
    return weakClassArr,aggClassEst

#利用训练出的多个弱分类器进行分类的函数
def adaClassify(datToClass,classifierArr):
    dataMatrix=mat(datToClass)#转换成矩阵
    m=shape(dataMatrix)[0]
    aggClassEst=mat(zeros((m,1)))
    for i in range(len(classifierArr)):#遍历所有的弱分类器
        classEst=stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst+=classifierArr[i]['alpha']*classEst
        print (aggClassEst)
    return sign(aggClassEst)

#ROC曲线绘制及AUC计算函数（曲线下的面积）
def plotRoc(predStrengths,classLabels):#第一个参数代表分类器的预测强度
    cur = (1.0,1.0) #绘制光标的位置
    ySum = 0.0 #用于计算AUC的值
    numPosClas = sum(array(classLabels)==1.0)
    yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort()#获取排好序的索引
    #构建画笔
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep;
        else:
            delX = xStep; delY = 0;
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print ('the Area Under the Curve is: ',ySum*xStep)

'''
datMat,classLabels=loadSimpData()
D=mat(ones((5,1))/5)
buildStump(datMat, classLabels, D)
classifierArray=adaBoostTrainDS(datMat, classLabels, 9)
'''
datArr,labelArr=loadDataSet('horseColicTraining2.txt')
classifierArray,aggClassEst=adaBoostTrainDS(datArr, labelArr, 10)
plotRoc(aggClassEst.T, labelArr)
'''
testArr,testlabelArr=loadDataSet('horseColicTest2.txt')
prediction10=adaClassify(testArr,classifierArray)
errArr=mat(ones((67,1)))
errArr[prediction10!=mat(testlabelArr).T].sum()
'''