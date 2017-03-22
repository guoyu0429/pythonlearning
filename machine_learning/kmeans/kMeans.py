#coding=utf-8
#k均值聚类
from numpy import *

#1.导入文本文件并解析
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat
#2.计算两个向量的欧氏距离（距离函数）
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)
#3.为给定数据集构建一个包含K个随机质心的集合
#随机质心要在整个数据集的边界之内，可以通过找到数据集每一维的最小和最大值来完成，然后生成0到1.0之间的随机数并通过取值范围和最小值，以便确保随机点在数据的边界之内
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))#create centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:,j]) 
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids
#4.k均值聚类算法
'''
创建K个点作为起始质心
当任意一个点的簇分配结果发生改变时
  对数据集中的没个数据点
      对每个质心
          计算质心与数据点之间的距离
    将数据点分配到距其最近的簇
对每个簇，计算簇中所有点的均值并将均值作为质心
'''
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):#数据集、簇的数目、计算距离的函数、创建初始簇心的函数
    m = shape(dataSet)[0]#确定数据集中数据点的总数
    clusterAssment = mat(zeros((m,2)))#创建一个矩阵用于存放每个点的簇分配结果：一列是记录簇索引值，一列存储误差（当前点到簇质心的距离）
    centroids = createCent(dataSet, k)#随机分配质心
    clusterChanged = True
    while clusterChanged:#当簇分类结果发生改变
        clusterChanged = False
        for i in range(m):#遍历每一个数据点
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])#计算点与质心的距离
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        print (centroids)
        for cent in range(k):#更新质心的位置
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#获得在这个簇中的所有点
            centroids[cent,:] = mean(ptsInClust, axis=0) #计算所有点的均值
    return centroids, clusterAssment

#5.提高聚类性能：二分k-均值算法
'''
将所有点看成一个簇
当簇数目小于k时
   对于每一个簇
        计算总误差
        在给定的簇上面进行k-均值聚类（k=2)
        计算将该簇一分为二的总误差
    选择使得误差最小的那个簇进行划分操作     
'''
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))#创建一个矩阵才存储分配结果和平方误差
    centroid0 = mean(dataSet, axis=0).tolist()[0]#计算整个数据集的质心
    centList =[centroid0] #用一个列表来保存质心
    for j in range(m):#遍历数据集所有点计算每个点到质心的误差值
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k):#当簇数目小于想要的簇数目时进行循环
        lowestSSE = inf#一开始将最小SSE设置为无穷大
        for i in range(len(centList)):#遍历簇列表中的每一个簇
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]#get the data points currently in cluster i
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:,1])#compare the SSE to the currrent minimum
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print ('sseSplit, and notSplit: ',sseSplit,sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        #更新簇的分配结果
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print ('the bestCentToSplit is: ',bestCentToSplit)
        print ('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids 
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
    return mat(centList), clusterAssment

#6.案例：对地图上的点进行聚类
import urllib.parse
import urllib.request
import json
#从Yahoo!返回一个字典
def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  #create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'#JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.parse.urlencode(params)#将创建的字典转换为可以通过URL进行传递的字符串格式
    yahooApi = apiStem + url_params      #print url_params
    print (yahooApi)
    c=urllib.request.urlopen(yahooApi)
    return json.loads(c.read())

from time import sleep
#将所有这些封装起来并且将相关信息保存到文件中
def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])#获取第2列和第3列的结果
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print ('%s\t%f\t%f' % (lineArr[0], lat, lng))
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else: print ('error fetching')
        sleep(1)
    fw.close()
#球面距离计算  
def distSLC(vecA, vecB):#Spherical Law of Cosines
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * \
                      cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0 #pi is imported with numpy

import matplotlib
import matplotlib.pyplot as plt
def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()

'''
#测试生成随机质心
datMat=mat(loadDataSet('testSet.txt'))
print(randCent(datMat,2))
'''
'''
#测试K均值聚类算法
datMat=mat(loadDataSet('testSet.txt'))
myCentroids,clustAssing=kMeans(datMat, 4)
print(myCentroids)
print(clustAssing)
'''
'''
#测试二分K均值算法
datMat=mat(loadDataSet('testSet2.txt'))
myCentroids,clustAssing=biKmeans(datMat, 3)
'''
clusterClubs(5)