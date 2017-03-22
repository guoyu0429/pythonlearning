#coding=utf-8
#决策树
from math import log
import operator
import treePlotter


#创建数据集
def createDataSet():
    dataSet=[[1,1,'yes'],
             [1,1,'yes'],
             [1,0,'no'],
             [0,1,'no'],
             [0,1,'no']
             ]
    labels=['no surfacing','flippers']
    return dataSet,labels

#计算给定数据集的熵    熵：H = -∑pi*log2pi
def calcShnnonEnt(dataSet):
    numEntries=len(dataSet)#数据集需要的是每一行一个训练样本，并且最后一列为labels
    labelCounts={}
    for featVec in dataSet:
        currentLabel=featVec[-1]#对每个样本来说，取其label就是每行的最后一个值
        #if currentLabel not in labelCounts.keys():  
        #   labelCounts[currentLabel] = 0  
        #labelCounts[currentLabel] += 1 
        labelCounts[currentLabel]=labelCounts.get(currentLabel,0)+1#等价于上面三行
    shannonEnt=0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries#选择该分类的概率值
        shannonEnt-=prob*log(prob,2)
    return shannonEnt
    
#按照给定特征划分数据集
"""
@brief 划分数据集 按照给定的特征划分数据集
@param[in] dataSet 待划分的数据集
@param[in] axis  划分数据集的特征
@param[in] value 需要返回的特征的值
@return retDataSet 返回划分后的数据集
"""
def splitDataSet(dataSet,axis,value):
    ''''' 
    axis：特征的坐标。axis=0时,第0个特征其值可能为0或1, 
    value=1时，dataSet前3个都符合，从而得到子集[[1,"yes"],[1,"yes"],[0,"no"]]。 
    '''
    retDataSet=[]#需要新创建一个列表变量，因列表的参数是按照引用方式传递的  
    for featVec in dataSet:
        #print(featVec)#[1, 1, 'yes']
        #print(featVec[axis])#给定的坐标值  1
        #print(value)#给定的值  1
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis]#取到axis之前，即第二位之前
            #print(reducedFeatVec)
            reducedFeatVec.extend(featVec[axis+1:])##加上从axis+1开始的东西
            retDataSet.append(reducedFeatVec)
    return retDataSet

#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    ''''' 
    数据集格式： 
    1.由列表元素组成的列表，并且所有列元素都要具有相同数据长度。 
    2.数据最后一列为label. 
    如同createDataSet()中dataSet变量的数据集 
    ''' 
    numFeatures=len(dataSet[0])-1#特征的个数，随便第一个实例的长度，去掉最后一个标签。  
    baseEntropy = calcShnnonEnt(dataSet)  #计算数据集的熵
    bestInfoGain = 0.0  #定义基尼指数  ，最大的信息增益
    bestFeature  = -1#最佳特征初始化
    for i in range(numFeatures):#对第I个特征进行处理
        #print(i)
        featList   = [example[i] for example in dataSet]#取第i列的特征赋值给feat_list  
        #print(featList)
        uniqueVals = set(featList)#集合唯一性，去除重复值 
        newEntropy = 0.0 
        for value in uniqueVals:
            subDataSet  = splitDataSet(dataSet, i, value)#对第i个特征，其值为value划分数据  
            prob        = len(subDataSet)/float(len(dataSet))  #计算分割后的数据集占总的数据的比
            #print(prob)
            newEntropy += prob*calcShnnonEnt(subDataSet) #调用计算熵的公式
        infoGain = baseEntropy - newEntropy#计算基尼指数。利用基尼指数得到熵最低的特征。
        if infoGain > bestInfoGain:  
            bestInfoGain = infoGain  
            bestFeature  = i  
    return bestFeature 

"""
@brief 计算一个特征数据列表中 出现次数最多的特征值以及次数
@param[in] 特征值列表
@return 返回次数最多的特征值
例如：[1,1,0,1,1]数据列表 返回 1
0"""
def majorityCnt(classList):             #classList为类标例表  
    classCount = {}  
    for vote in classList:  
        classCount[vote] = classCount.get(vote,0)+1  
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)  
    return sortedClassCount[0][0] 

"""
@brief 主程序，递归产生决策树。 
params: 
dataSet:用于构建树的数据集,最开始就是data_full，然后随着划分的进行越来越小，第一次划分之前是17个瓜的数据在根节点，然后选择第一个bestFeat是纹理 
纹理的取值有清晰、模糊、稍糊三种，将瓜分成了清晰（9个），稍糊（5个），模糊（3个）,这个时候应该将划分的类别减少1以便于下次划分 
labels：还剩下的用于划分的类别 
data_full：全部的数据 
label_full:全部的类别 

既然是递归的构造树，当然就需要终止条件，终止条件有三个： 
1、当前节点包含的样本全部属于同一类别；-----------------注释1就是这种情形 
2、当前属性集为空，即所有可以用来划分的属性全部用完了，这个时候当前节点还存在不同的类别没有分开，这个时候我们需要将当前节点作为叶子节点， 
同时根据此时剩下的样本中的多数类（无论几类取数量最多的类）-------------------------注释2就是这种情形 
3、当前节点所包含的样本集合为空。比如在某个节点，我们还有10个西瓜，用大小作为特征来划分，分为大中小三类，10个西瓜8大2小，因为训练集生成 
树的时候不包含大小为中的样本，那么划分出来的决策树在碰到大小为中的西瓜（视为未登录的样本）就会将父节点的8大2小作为先验同时将该中西瓜的 
大小属性视作大来处理。 
构
"""
def createTree(dataSet, labels):                            #labels  = ["no surfacing","flippers"],其值为特征名  
    classList = [example[-1] for example in dataSet]            #取dataSet每个实例的最后一个元素，也即label  
    if classList.count(classList[0]) == len(classList):         #类别完全相同则停止划分，取第一个就行了，第一个的个数等于所有的个数  
        return classList[0]  
    if len(dataSet[0]) == 1:                                #遍历完所有特征时，返回label出现次数最多的  
        return majorityCnt(classList)  
    bestFeat      = chooseBestFeatureToSplit(dataSet)  
    bestFeatLabel = labels[bestFeat]                        #labels列表中包含了具体的属性值  
    myTree        = {bestFeatLabel:{}}                      #关键所在  
    del(labels[bestFeat])                                   #类标的某个特征被选择了后，就不必再考虑了  
    featValues = [example[bestFeat] for example in dataSet]  
    uniqueVals = set(featValues)  
    for value in uniqueVals:  
        subLabels = labels[:]  
        #递归调用createTree()函数，并且将返回的tree插入到myTree字典中，  
        #利用最好的特征划分的子集作为新的dataSet传入到createTree()函数中。  
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)  
    return myTree

#决策树的分类函数
def classify(inputTree,featLabels,testVec): 
    firstStr=list(inputTree.keys())[0]
    secondDict=inputTree[firstStr]
    #将标签字段转换为索引
    featIndex=featLabels.index(firstStr)
    for key in secondDict.keys():
        if  testVec[featIndex]==key:
            #如果到达叶子节点，返回标签值
                if type(secondDict[key]).__name__=='dict':
                  classLabel=classify(secondDict[key],featLabels.testVec)
                else:
                    classLabel=secondDict[key]
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw=open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr=open(filename)
    return pickle.load(fr)
  
"""
myDat,labels = createDataSet()
myTree=treePlotter.retrieveTree(0)
print(classify(myTree, labels, {1,0}))
print(classify(myTree, labels, {1,1}))
"""
fr=open('lenses.txt')
lenses=[inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels=['age','prescript','astigmatic','tearRate']
lensesTree=createTree(lenses, lensesLabels)
treePlotter.createPlot(lensesTree)

