#coding=utf-8
#回归
'''
总结：
回归是用来预测连续型变量的，
（1）给定输入矩阵，如果矩阵的逆存在的话，回归法都可以用
（2)当矩阵的逆不能直接计算或者样本数比特征数多时，可以考虑使用岭回归，岭回归是缩减法的一种，相当于对回归系数的大小施加了限制
（3）另一种缩减法是lasso，lasso难以求解，但可以使用计算机岸边的逐步线性回归法来求得近似结果
（4）偏差方差折中概念可以帮助我们对现有模型进行改进
'''
from numpy import *
import matplotlib.pyplot as plt

#标准回归函数和数据导入函数
def loadDataSet(filename):
    numFeat=len(open(filename).readline().split('\t'))-1
    dataMat=[]
    labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#计算最佳拟合曲线
def standRegres(xArr,yArr):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    xTx=xMat.T*xMat
    if linalg.det(xTx)==0.0:#该函数用来计算行列式，如果行列式为0，计算逆矩阵的时候将出现错误
        print('this matrix is singular,cannot do inverse')
        return
    ws=xTx.I *(xMat.T*yMat)
    return ws#返回的就是回归系数

#局部加权线性回归函数
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    m=shape(xMat)[0]
    weights=mat(eye((m)))
    for j in range(m):#计算每个样本点对应的权重值，随着样本点与待预测点距离的递增，权重将以指数级衰减，输入参数k控制衰减的速度
        diffMat=testPoint-xMat[j,:]
        weights[j,j]=exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx=xMat.T*(weights*xMat)
    if linalg.det(xTx)==0.0:
        print('this matrix is singular,cannot do inverse')
        return 
    ws=xTx.I*(xMat.T*(weights*yMat))
    return testPoint*ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    m=shape(testArr)[0]
    yHat=zeros(m)
    for i in range(m):
        yHat[i]=lwlr(testArr[i],xArr,yArr,k)
    return yHat

#计算预测误差
def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()

#岭回归：解决样本特征多余样本数的问题
#用户计算回归系数
def ridgeRegres(xMat,yMat,lam=0.2):
    xMat=mat(xMat)
    yMat=mat(yMat).T
    xTx=xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam#eye函数构建单位矩阵
    if linalg.det(denom) == 0.0:
        print ('This matrix is singular, cannot do inverse')
        return
    ws = denom.I * (xMat.T*yMat)
    return ws

#用户在一组数据上进行测试
def ridgeTest(xArr,yArr):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    #数据标准化处理：
    yMat = yMat - yMean     #to eliminate X0 take mean off of Y
    #regularize X's
    xMeans = mean(xMat,0)   #calc mean then subtract it off
    xVar = var(xMat,0)      #calc variance of Xi then divide by it
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat

def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat

#前向逐步回归:每一步都尽可能减少误差，一开始，所有权重设为1，然后每一步所做的决策是对某个权重增加或减少一个很小的值
def stageWise(xArr,yArr,eps=0.01,numIt=100):#输入数据xArr，预测变量yArr，eps每次迭代需要调整的步长，numIt迭代次数
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean     #can also regularize ys but will get smaller coef
    xMat = regularize(xMat)
    m,n=shape(xMat)
    #returnMat = zeros((numIt,n)) #testing code remove
    ws = zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):
        print (ws.T)
        lowestError = inf; 
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        #returnMat[i,:]=ws.T
    #return returnMat


'''
xArr,yArr=loadDataSet('ex0.txt')
print(xArr[0:2])
ws=standRegres(xArr, yArr)
print(ws)
xMat=mat(xArr)
yMat=mat(yArr)
yHat=xMat*ws#预测出的y
print(corrcoef(yHat.T,yMat))#计算预测值和真实值之间的相关系数

#绘制数据集散点图和最佳拟合直线图
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
xCopy=xMat.copy()
xCopy.sort(0)
yHat=xCopy*ws
ax.plot(xCopy[:,1],yHat)
plt.show()
'''

'''
#测试局部加权函数
xArr,yArr=loadDataSet('ex0.txt')
yHat=lwlrTest(xArr, xArr, yArr, 0.003)#k=1.0时模型效果与最小二乘法差不多，k=0.01时该模型可以挖出数据的潜在规律，k=0.003时则考虑了太多的噪声，导致了过拟合现象
xMat=mat(xArr)
strInd=xMat[:,1].argsort(0)
xSort=xMat[strInd][:,0,:]
fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(xSort[:,1],yHat[strInd])
ax.scatter(xMat[:,1].flatten().A[0],mat(yArr).T.flatten().A[0],s=2,c='red')
plt.show()
'''

'''
#示例：预测鲍鱼额年龄
abX,abY=loadDataSet('abalone.txt')
yHat01=lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
yHat1=lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
yHat10=lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
print(rssError(abY[0:99], yHat01.T))
print(rssError(abY[0:99], yHat1.T))
print(rssError(abY[0:99], yHat10.T))
'''
'''
#测试岭回归
abX,abY=loadDataSet('abalone.txt')
ridgeWeights=ridgeRegres(abX, abY)
fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(ridgeWeights)
plt.show()
'''

#测试前向逐步回归
xArr,yArr=loadDataSet('abalone.txt')
stageWise(xArr, yArr, 0.01, 200)