#coding=utf-8
#绘制属树形图
import matplotlib.pyplot as plt

#定义文本框和箭头格式
decisionNode=dict(boxstyle="sawtooth",fc="0.8")#设置文本框形状，前面的参数代表框的形状，后面的参数代表边框背景灰度，值从0-1,0为黑，1为白
leafNode=dict(boxstyle="round4",fc="0.8")
arrow_args=dict(arrowstyle="<-")#设置箭头的形状

#绘制带箭头的注解
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    #matplotlib.pyplot模块提供的一个注解函数，可以用来对坐标中的数据进行注解
    #xy代表点的坐标，xycoords是坐标xy的说明，xytext是注解内容的位置坐标，textcoorda是注解内容额说明，arrowprops是箭头的形状，bbox为文本边框形状
    createPlot.axl.annotate(nodeTxt,xy=parentPt ,xycoords='axes fraction',xytext=centerPt,textcoords='axes fraction',va="center",ha="center",bbox=nodeType,arrowprops=arrow_args)

#创建新图形并清空绘图区   ，然后再绘图区上绘制两个代表不同类型的树节点
#1.主函数
def createPlot(inTree):
    fig=plt.figure(1,facecolor='white')
    fig.clf()
    axprop=dict(xticks=[],yticks=[])
    createPlot.axl=plt.subplot(111,frameon=False,**axprop)
    plotTree.totalW=float(getNumLeafs(inTree))#树的宽度
    plotTree.totalD=float(getTreeDepth(inTree))#树的深度
    plotTree.xOff=-0.5/plotTree.totalW;#追踪已经绘制的节点位置，以及放置下一个节点的位置
    plotTree.yOff=1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()

#获得叶子节点的数目
def getNumLeafs(myTree):
    numLeafs=0
    firstStr=list(myTree.keys())[0]#从第一个关键字出发
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        #判断子节点是否为字典类型（代表下面还有值）
        if type(secondDict[key]).__name__=='dict':
            #递归调用getNumLeafs函数
            numLeafs+=getNumLeafs(secondDict[key])
        else:
            #否则是叶子节点，总数加1
            numLeafs+=1
    return numLeafs

#获取树的深度
def getTreeDepth(myTree):
    maxDepth=0
    firstStr=list(myTree.keys())[0]
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth=1+getTreeDepth(secondDict[key])
        else:
            thisDepth=1
        if thisDepth>maxDepth:
            maxDepth=thisDepth
    return maxDepth

#输出预先存储的树信息
def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},  
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}  
                  ] 
    return listOfTrees[i]

#3.在父子节点间填充文本信息
def plotMidText(cntrPt,parentPt,txtString):
    xMid=(parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid=(parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.axl.text(xMid,yMid,txtString)

#2.画树
def plotTree(myTree,parentPt,nodeTxt):
    #计算树宽与高
    numLeafs=getNumLeafs(myTree)
    depth=getTreeDepth(myTree)
    firstStr=list(myTree.keys())[0]
    #接下来就是计算cntrPt，就是节点的中间位置  
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)  
    #绘出子节点具有的特征值
    plotMidText(cntrPt, parentPt, nodeTxt) #计算父节点和子节点的中间位置 ，并在此处添加文本信息
    plotNode(firstStr, cntrPt, parentPt,decisionNode)
    secondDict=myTree[firstStr]
    #按比例递减y坐标值
    plotTree.yOff=plotTree.yOff-1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            #如果是叶子节点，则画出叶子节点
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            #不是叶子节点的话，递归调用，在画了所有叶子节点后，增加Y偏移
            plotTree.xOff=plotTree.xOff+1.0/plotTree.totalW
        plotNode(secondDict[key], (plotTree.xOff,plotTree.yOff), cntrPt,leafNode)
        plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
    plotTree.yOff=plotTree.yOff+1.0/plotTree.totalD
            
    
    
    
"""   
myTree=retrieveTree(0)
createPlot(myTree)
"""