#coding=utf-8
#树回归
import  numpy as np


#1.读取文件，然后将每一行的内容保存成一组浮点数
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine =list(map(float,curLine)) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat
 
#2.在给定特征和特征值的情况下，通过数组过滤方式将上述数据集合切分得到两个子集并返回   
def binSplitDataSet(dataSet, feature, value):  
    mat0 = dataSet[np.nonzero(dataSet[:,feature] > value)[0],:] 
    mat1 = dataSet[np.nonzero(dataSet[:,feature] <= value)[0],:] 
    return mat0,mat1 
#3.创建叶节点函数
def regLeaf(dataSet):#returns the value used for each leaf
    return np.mean(dataSet[:,-1])
#4.总方差计算
def regErr(dataSet):
    return np.var(dataSet[:,-1]) * np.shape(dataSet)[0]#var是均方差函数*数据集中样本的个数，最后返回总方差

#5.用最佳方式切分数据集和生成相应叶节点
#相当于采用了预剪枝技术来说避免过拟合技术
#不足：对输入参数tolS和tolN非常敏感，如果使用其他值可能得不到很多的效果
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tols=ops[0]#允许的误差下降值
    tolN=ops[1]#切分的最小样本数
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #exit cond 1
        return None, leafType(dataSet)
    m,n=np.shape(dataSet)
    S=errType(dataSet)
    bestS=np.Inf;bestIndex=0;bestValue=0
    for featIndex in range(n-1):
        for splitVal in set((dataSet[:,featIndex].T.A.tolist())[0]): 
            mat0,mat1=binSplitDataSet(dataSet, featIndex, splitVal)#将数据集切分成两份
            if(np.shape(mat0)[0]<tolN) or (np.shape(mat1)[0]<tolN):continue
            newS=errType(mat0)+errType(mat1)#计算切分的误差
            if newS<bestS:#如果当前误差《当前最小误差，那么将当前切分设定为最佳切分并更新最小误差
                bestIndex=featIndex
                bestValue=splitVal
                bestS=newS
    if(S-bestS)<tols:#如果误差减小不大则退出
        return None,leafType(dataSet)
    mat0,mat1=binSplitDataSet(dataSet, bestIndex, bestValue)
    if((np.shape(mat0))[0]<tolN) or (np.shape(mat1)[0]<tolN):#如果切分出的数据集很小则退出
        return None,leafType(dataSet)
    return bestIndex,bestValue

#6.树构建函数
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):#assume dataSet is NumPy Mat so we can array filtering
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)#choose the best split
    if feat == None: return val #if the splitting hit a stop condition return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree 

#7.后剪枝技术：将数据集分成测试集合训练集
'''
基于已有的树切分测试数据
    如果存在任一子集是一棵树，则在该子集递归剪枝过程
  计算将当前两个叶节点合并后的误差
 计算不合并的误差
 如果合并后会降低误差的话，就将节点合并
'''
#测试输入变量是否是一棵树，返回布尔类型的结果（判断当前处理的节点是否是叶节点）
def isTree(obj):
    return (type(obj).__name__=='dict')
#递归函数：从上往下便利树直到叶节点为止，如果有两个叶节点则计算它们的平均值
def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0
#剪枝函数
def prune(tree, testData):#待剪枝的树和剪枝所需的测试数据
    if np.shape(testData)[0] == 0: return getMean(tree) #首先确认测试集是否为空
    if (isTree(tree['right']) or isTree(tree['left'])):#如果是字数调用函数prune进行剪枝
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet)
    #if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(np.power(lSet[:,-1] - tree['left'],2)) +\
            sum(np.power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(np.power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge: 
            print ('merging')
            return treeMean
        else: return tree
    else: return tree

#8.模型树：在叶节点生成线性模型而不是常数值

#主要功能是：数据集格式化成目标变量Y和目标变量X
def linearSolve(dataSet):   #helper function used in two places
    m,n = np.shape(dataSet)
    X = np.mat(np.ones((m,n))); Y = np.mat(np.ones((m,1)))#create a copy of data with 1 in 0th postion
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]#and strip out Y
    xTx = X.T*X
    if np.linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y
#当数据不再需要切分的时候负责生成叶节点
def modelLeaf(dataSet):#create linear model and return coeficients
    ws,X,Y = linearSolve(dataSet)
    return ws
#在给定的数据集上计算误差
def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(np.power(Y - yHat,2))

#9.用树回归进行预测
#对回归树叶节点进行预测的函数   
def regTreeEval(model, inDat):
    return float(model)
#对模型树叶节点数据进行预测
def modelTreeEval(model, inDat):
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)
#自顶向下遍历整棵树，直到命中叶节点
def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']): return treeForeCast(tree['left'], inData, modelEval)
        else: return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']): return treeForeCast(tree['right'], inData, modelEval)
        else: return modelEval(tree['right'], inData)
        
def createForeCast(tree, testData, modelEval=regTreeEval):
    m=len(testData)
    yHat = np.mat(np.zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
    return yHat

'''
#测试预剪枝
myDat=loadataSet('ex00.txt')
myMat=mat(myDat)
createTree(myMat)
'''
'''
#测试后剪枝
myDat2=loadDataSet('ex2.txt')
myMat2=mat(myDat2)
myTree=createTree(myMat2,ops=(0,1))
myMatTest=loadDataSet('ex2test.txt')
myMat2Test=mat(myMatTest)
prune(myTree,myMat2Test)
'''
'''
#测试模型树
myMat2=mat(loadDataSet('exp2.txt'))
createTree(myMat2,modelLeaf,modelErr,(1,10))
'''
#测试树回归进行预测
#创建回归树
'''
trainMat=mat(loadDataSet('bikeSpeedVsIq_train.txt'))
testMat=mat(loadDataSet('bikeSpeedVsIq_test.txt'))
myTree=createTree(trainMat,ops=(1,20))
yHat=createForeCast(myTree, testMat[:,0])
corrcoef(yHat, testMat[:,1], rowvar=0)[0,1]
#创建模型树
myTree=createTree(trainMat,modelLeaf,modelErr,ops=(1,20))
yHat=createForeCast(myTree, testMat[:,0],modelTreeEval)
corrcoef(yHat, testMat[:,1], rowvar=0)[0,1]
'''