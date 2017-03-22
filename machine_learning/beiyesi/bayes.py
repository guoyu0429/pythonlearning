#coding=utf-8
#案例一：区分垃圾邮件
#词表到向量的转换函数
from numpy import *
import re

def loadDataSet():
    postingList=[['my','dog','has','flea',\
                  'problems','help','please'],
                 ['maybe','not','take','him',\
                  'to','dog','park','stupid'],
                 ['my','dalmation','is','so','cute',\
                  'I','love','him'],
                 ['stop','posting','stupid','worthless','garbage'],
                 ['mr','licks','ate','my','steak','how',\
                  'to','stop','him'],
                 ['quit','buying','worthless','dog','food','stupid']
                 ]
    classVec=[0,1,0,1,0,1]#1 代表侮辱性文字  0 代表正常言论
    return postingList,classVec  #返回进行词条切分后的文档集合     类别标签的集合（人工标注）


#文本解析:接受一个大写字符串并将其解析为字符串列表。该函数去掉少于两个字符的字符串，并将所有字符串准换成小写。
def textParse(bigString):
    listOfTokens=re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>2]
#对贝叶斯垃圾邮件分类器进行自动化处理
def spamTest():
    docList=[]
    classList=[]
    fullText=[]
    #导入并解析文本文件
    for i in range(1,26):
        wordList=textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList=textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        classList.append(0)
    vocabList=createVocabList(docList)
    trainingSet=list(range(50))#总共50封邮件
    testSet=[]
    #随机构建训练集
    for i in range(10):
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]
    trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(setOfWord2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam=trainNBO(array(trainMat), array(trainClasses))
    errorCount=0
    #对测试集分类
    for docIndex in testSet:
        wordVector=setOfWord2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam)!=classList[docIndex]:
            errorCount+=1
        print ('the error rate is :',float(errorCount)/len(testSet))
#创建一个包含在所有文档中出现的不重复词的列表（即获得词汇表）
def createVocabList(dataSet):
    vocabSet=set([]) #创建一个空集
    for document in dataSet:
        vocabSet=vocabSet | set(document) #创建两个集合的并集
    return list(vocabSet)

#输入词汇表及某个文档，输出文档向量，向量中每一元素为0或1，分别表示词汇表中的单词是否在输入文档中出现
def setOfWord2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)#创建一个和词汇表等长的向量，将其元素设为0
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:
            print ('the world:%s is not in my Vocablary' %word)
    return returnVec

#朴素贝叶斯词袋模型  功能类似于setOfWord2Vec：不同点在于每当遇到一个单词时，会增加词向量中的对应值，而不只是将对应的数值设为1 
def bagOfWords2VecMN(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
    return returnVec

#朴素贝叶斯分类器训练函数
#输入参数为文档矩阵trainMatrix,由每篇文档类别标签所构成的向量
def trainNBO(trainMatrix,trainCatagory):
    numTrainDocs=len(trainMatrix)
    numWords=len(trainMatrix[0])
    pAbusive=sum(trainCatagory)/float(numTrainDocs)
    p0Num=ones(numWords)
    p1Num=ones(numWords)
    p0Denom=2.0
    p1Denom=2.0
    for i in range(numTrainDocs):
        if trainCatagory[i]==1:
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    p1Vect=log(p1Num/p1Denom)#采用对数解决下溢出问题（因乘数太小而出现程序溢出问题）
    p0Vect=log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

#朴素贝叶斯分类函数
#vec2Classify 输入的 要分类的向量    p0Vec,p1Vec,pClass1：使用trainNBO计算得到的概率
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1=sum(vec2Classify*p1Vec)+log(pClass1)
    p0=sum(vec2Classify*p0Vec)+log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0
 
 
#遍历函数，将所有的操作封装起来   
def testingNB():
    listOPosts,listClasses=loadDataSet()
    myVocabList=createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWord2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb=trainNBO(array(trainMat),array(listClasses))
    testEntry=['love','my','dalmation']
    thisDoc=array(setOfWord2Vec(myVocabList,testEntry))
    print (testEntry,'classified as :',classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry=['stupid','garbage']
    thisDoc=array(setOfWord2Vec(myVocabList,testEntry))
    print (testEntry,'classified as :',classifyNB(thisDoc, p0V, p1V, pAb))
    
 
 
    
#testingNB()
#spamTest()


