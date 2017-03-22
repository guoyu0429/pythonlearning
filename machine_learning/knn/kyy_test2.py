#coding=utf-8
#使用k-近邻算法改进约会网站的配对效果
#数据文件放在datingTestSet.txt中，被"\t"隔开为4列，有3种特征：
import codecs
from numpy import *
import operator  
import matplotlib.pyplot as plt

#1.准备数据：需要将数据转换为后续方便处理的格式，如特征用一个变量表示，目标变量用另一个变量表示，并将枚举类型的目标类别用数字代替。
#读入数据，输出训练样本和类别标签向量。
def file2matrix(filename):
    with codecs.open(filename) as f:#打开文件
        arrayOLines=f.readlines()#读取文件
    numberOflines=len(arrayOLines)#计算文件的行数（此数据中总共1000行）
    returnMat=zeros((numberOflines,3))#产生1000x3的0矩阵
    classLabelVector=[]#生成一个序列，主要操作是切片，用于保存从数据文件中读取的每一行的分类标签
    index=0
    like_type={}
    like_type['largeDoses']=3#代表极具魅力的人
    like_type['smallDoses']=2#代表魅力一般的人
    like_type['didntLike']=1#代表不喜欢的人
    for line in arrayOLines:#遍历每一行
        line=line.strip()#去掉空格
        line=line.strip("\n")
        listFromLine=line.split("\t")#根据\t进行分隔
        returnMat[index,:]=listFromLine[0:3]#取前3个切片放到index行
        if listFromLine[-1] in like_type:#如果最后一列在type取值当中
            classLabelVector.append(int(like_type[listFromLine[-1]]))#获取分类标签
        else:
            print ("like_type error",listFromLine[-1])
            classLabelVector.append(0)
        index+=1
    return returnMat,classLabelVector#返回特征矩阵和分类矩阵

#2.分析数据-数据可视化
def showDateSet(datingDataMat, datingLabels):    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:,0], datingDataMat[:,1], c = 15*array(datingLabels), s = 15*array(datingLabels), label=u'散点图')
    plt.legend(loc = 'upper left')
    plt.xlabel(u"玩视频游戏所耗得时间比")
    plt.ylabel(u"每年获取的飞行常客里程数")
    plt.show()

#3.数据归一化处理   newValue = (oldValue - min)/(max - min)
def autoNorm(dataSet):
    minVals=dataSet.min(0)#返回每列的最小值
    maxVals=dataSet.max(0)#返回每列的最大值
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))# dataSet所有元素都变为0,即同dataSet数组一样大小的全零矩阵  
    m=dataSet.shape[0]#返回行数
    normDataSet = dataSet - tile(minVals,(m,1))   #tile(minVals,(m,1))：将minVals复制为m*1份  
    normDataSet = normDataSet/(tile(ranges,(m,1)))#在numpy包中，矩阵除法需要使用函数linalg.solve(matA,matB)  
    return normDataSet, ranges, minVals  

#4.计算距离
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
        voteIlabel=labels[sortedDistIndicies[i]]#获取前K个标签
        classCount[voteIlabel]=classCount.get(voteIlabel)#先用get()函数将label I在字典里默认值为0,若无直接+1,也即label I的投票+1。
        sortedClassCount=sorted(classCount.items())
    return sortedClassCount[0][0]#返回出现次数最多标签的标签类

#5.测试算法：使用错误率来检验我们的分类器的性能
def datingClassTest():
    hoRatio=0.10#10%的数据作为测试
    datingDataMat,datingLabels=file2matrix("datingTestSet.txt") #得到训练样本和标签向量
    normMat,ranges,minVals=autoNorm(datingDataMat)#数据归一化
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)#获得前10%作为测试数据
    errorCount=0.0#错误率初始化
    for i in range(numTestVecs):
         #前10%行的数据作为测试集，并且对测试集中的每一行都进行预测，对比测试集中实际的label  
        #后90%行的数据全部作为训练集，每个测试集样本都要跟90%的训练集计算距离，算出最相似的label
        classifierResult =classify(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print('the classifier came back with: %d ,the real answer is: %d' % (classifierResult, datingLabels[i]))  
        if (classifierResult != datingLabels[i]): 
            errorCount += 1.0
    print('the totle error rate is : %f' % (errorCount/float(numTestVecs)))
    
#使用算法
def classifyPerson():
    resultList  = ["not at all","in small doses","in large doses"]  
    percentTats = float(input("percentage of time spent playing video games? "))  
    ffMiles     = float(input("frequent flier miles earned per year? "))  
    iceCream    = float(input("liters of ice cream consumed per year? "))  
    inArr       = array([ffMiles, percentTats, iceCream])   
    datingDataMat, datingLabels = file2matrix("datingTestSet.txt")  
    normMat,ranges,minVals = autoNorm(datingDataMat) #需要对新来的测试集也做归一化，故需要用到ranges和minVals两个变量  
    classifierResult       = classify((inArr-minVals)/ranges, normMat, datingLabels, 3)  
    print ('You will probably like this person: ',resultList[classifierResult - 1])
    
#classifyPerson()
print(datingClassTest())
