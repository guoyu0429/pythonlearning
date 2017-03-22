#coding=utf-8
#apriori算法:频繁项集和关联规则

#1.频繁项集（满足最小支持度要求）
#创建一个用于测试的简单数据集
def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
#构建集合C1，C1是大小为1的所有候选项集的集合
'''
对数据集中的每条交易记录tran
对每个候选项集can
    检查一下can是否是tran的子集
    如果是，则增加can的计数值
    对每个候选项集
    如果其支持度不低于最小值，则保留该项集
    返回所有频繁项集列表
'''
def createC1(dataSet):
    c1=[]#用于存储所有不重复的项值
    for transaction in dataSet:
        for item in transaction:
            if not [item] in c1:
                c1.append([item])
    c1.sort()
    return map(frozenset,c1)#frozenset是冻结的集合，是不可变的，存在哈希值
#该函数用于从c1生成L1
def scanD(D,Ck,minSupport):#数据集ck,包含候选集合的列表、最小支持度
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not can in ssCnt: ssCnt[can]=1
                else: ssCnt[can] += 1
    numItems=float(len(D))
    retList=[]
    supportData={}
    for key in ssCnt:
        support=ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key]=support
    return retList,supportData

#组织完整的Apriori算法
'''
当集合中项的个数大于0时：
    构建一个k个项组成的候选项集的列表
    当检查数据已确认每个项集都是频繁的
    保留频繁项集并构建k+1项组成的候选项集的列表
'''
#创建候选项集ck
def aprioriGen(Lk, k): #频繁项集列表lk，项集元素个数k，输出为ck
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):#取lk中的每一个元素和剩下的其他元素
        for j in range(i+1, lenLk): 
            #如果这两个集合的前面k-2个元素都相等，那么就将这两个集合合成一个大小为k的集合
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1==L2: #if first k-2 elements are equal
                retList.append(Lk[i] | Lk[j]) #set union
    return retList

def apriori(dataSet, minSupport = 0.5):
    C1 = list(createC1(dataSet))
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)#scan DB to get Lk
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

#2.关联规则（可信度）
#主函数，可调用下面两个函数
def generateRules(L, supportData, minConf=0.7):  #频繁项集列表、包含频繁项集支持数据的字典、最小可信度阈值
    bigRuleList = []#包含可信度的规则列表
    for i in range(1, len(L)):#only get the sets with two or more items
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList         
#对规则进行评估
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = [] #规则
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq] #calc confidence
        if conf >= minConf: 
            print (freqSet-conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH
#生成候选规则
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)): #try further merging
        Hmp1 = aprioriGen(H, m+1)#create Hm+1 new candidates
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):    #need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


#3.示例1：在美国国会投票记录中发现关联规则
from time import sleep
from votesmart import votesmart
votesmart.apikey = 'a7fa40adec6f4a77178799fae4441030'
#收集美国国会议案中actionID的函数
def getActionIds():
    actionIdList = []; billTitleList = []
    fr = open('recent20bills.txt') 
    for line in fr.readlines():
        billNum = int(line.split('\t')[0])
        try:
            billDetail = votesmart.votes.getBill(billNum) 
            for action in billDetail.actions:
                #过滤出包含投票的行为
                if action.level == 'House' and \
                (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
                    actionId = int(action.actionId)
                    print ('bill: %d has actionId: %d' % (billNum, actionId))
                    actionIdList.append(actionId)
                    billTitleList.append(line.strip().split('\t')[1])
        except:
            print ('problem getting bill %d' % billNum)
        sleep(1)            #为礼貌访问网站而做些延迟                         
    return actionIdList, billTitleList
#基于投票数据的事物列表填充函数
def getTransList(actionIdList, billTitleList): #this will return a list of lists containing ints
    itemMeaning = ['Republican', 'Democratic']#list of what each item stands for
    for billTitle in billTitleList:#fill up itemMeaning list
        itemMeaning.append('%s -- Nay' % billTitle)
        itemMeaning.append('%s -- Yea' % billTitle)
    transDict = {}#list of items in each transaction (politician) 
    voteCount = 2
    for actionId in actionIdList:
        sleep(3)
        print ('getting votes for actionId: %d' % actionId)
        try:
            voteList = votesmart.votes.getBillActionVotes(actionId)
            for vote in voteList:
                if not transDict.has_key(vote.candidateName): 
                    transDict[vote.candidateName] = []
                    if vote.officeParties == 'Democratic':
                        transDict[vote.candidateName].append(1)
                    elif vote.officeParties == 'Republican':
                        transDict[vote.candidateName].append(0)
                if vote.action == 'Nay':
                    transDict[vote.candidateName].append(voteCount)
                elif vote.action == 'Yea':
                    transDict[vote.candidateName].append(voteCount + 1)
        except: 
            print ('problem getting actionId: %d' % actionId)
        voteCount += 2
    return transDict, itemMeaning
