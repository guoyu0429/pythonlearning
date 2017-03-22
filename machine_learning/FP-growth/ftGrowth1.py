#coding=utf-8
#ft-growth：构建FP树，然后利用它来挖掘频繁项集。


#1.创建一个类来保存树的每一个节点
class treeNode:
    #节点的名字、计数值、父节点
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None  #链接相似的元素项
        self.parent = parentNode      #needs to be updated
        self.children = {}   #存放节点的子节点
    #给count变量增加给定值
    def inc(self, numOccur):
        self.count += numOccur
    #将树以文本形式显示 
    def disp(self, ind=1):
        print ('  '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)

#2.FP树构建函数
#使用数据集及最小支持度作为参数构建FP树。构建的过程会遍历数据集两次
#第一次遍历数据集并统计每个元素项出现的频度
def createTree(dataSet,minSup=1):
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item]=headerTable.get(item,0)+dataSet[trans]
    for k in list(headerTable.keys()):
        #移除不满足最小支持度的元素项
        if headerTable[k]<minSup:
            del(headerTable[k])
    freqItemSet=set(headerTable.keys())
    #如果没有元素项满足要走，则退出
    if len(freqItemSet)==0:return None,None
    for k in headerTable:
        headerTable[k]=[headerTable[k],None]#对头指针稍加扩展以便可以保存计数值及指向每种类型第一个元素相的指针
    retTree=treeNode('Null Set',1,None)#创建只包含空集合的根节点
    for tranSet,count in dataSet.items():#再一次比那里数据及
        localD={}
        #根据全局频率对每个事务中的元素进行排序
        for item in tranSet:
            if item in freqItemSet:
                localD[item]=headerTable[item][0]
        if len(localD)>0:
            orderItems=[v[0] for v in sorted(localD.items(),key=lambda p:p[1],reverse=True)]
            #使用排序后的频率项集对树进行填充
            updateTree(orderItems,retTree,headerTable,count)
    return retTree,headerTable

def updateTree(items,inTree,headerTable,count):
    if items[0] in inTree.children:#测试事务中的第一个元素项是否作为子节点存在
        inTree.children[items[0]].inc(count)#如果存在的话更新计数项
    else:
        inTree.children[items[0]]=treeNode(items[0],count,inTree)#不存在的话创建一个treeNode并添加到树中
        if headerTable[items[0]][1]==None:#更新头指针表
            headerTable[items[0]][1]=inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items)>1:
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)
        
def updateHeader(nodeToTest, targetNode):   
    while (nodeToTest.nodeLink != None):    
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode
    
#3.从FP树中挖掘频繁项集：通过查找元素项的条件基
def ascendTree(leafNode, prefixPath): #ascends from leaf node to root
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)
    
def findPrefixPath(basePat, treeNode): #treeNode comes from header table
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1: 
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats

#递归查找频繁项集
#条件模式基：以所查找元素项为结尾的路径集合
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1])]
    for basePat in bigL:  #从头指针表的底端开始
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        #print 'finalFrequent Item: ',newFreqSet    #append to set
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        #print 'condPattBases :',basePat, condPattBases
        #2. 从条件模式基来构建FP树
        myCondTree, myHead = createTree(condPattBases, minSup)
        #print 'head from conditional tree: ', myHead
        if myHead != None: #3. 挖掘条件FP树
            #print 'conditional tree for: ',newFreqSet
            #myCondTree.disp(1)            
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)
    
#简单数据集及数据包装器
def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict
    
'''
rootNode=treeNode('pyramid',9,None)
rootNode.children['eye']=treeNode('eye',13,None)
rootNode.children['phoenix']=treeNode('phoenix',3,None)
rootNode.disp()
'''
'''
simpDat=loadSimpDat()
initSet=createInitSet(simpDat)
myFPtree,myHeaderTab=createTree(initSet,3)
myFPtree.disp()
myFreqList = []
mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
'''
#从新闻网站点击流中挖掘
minSup = 3
parsedDat=[line.split() for line in open('kosarak.dat').readlines()]
initSet=createInitSet(parsedDat)
myFPtree,myHeaderTab=createTree(initSet, 100000)
myFreqList = []
mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
print(len(myFreqList))