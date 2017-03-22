#coding=utf-8
'''
SVD:奇异值分解，可用于降维，一般可使维度高于总能量的90%
利用SVD来逼近矩阵并从中提取重要特征，通过保留80%~90%的能量，就可以得到重要的特征并去掉噪声
'''
from numpy import *
from numpy import linalg as la

def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]

def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]
    
#基于协同过滤的推荐引擎
#1.相似度计算
#欧式距离
def ecludSim(inA,inB):
    return 1.0/(1.0+la.norm(inA,inB))
#皮尔逊系数
def pearsSim(inA,inB):
    if len(inA)<3:#如不存在3个点或更多的点，返回1.0，说明此时两个向量完全相关
        return 1.0
    return 0.5+0.5*corrcoef(inA, inB, rowvar=0)[0][1]
#余弦相似度
def cosSim(inA,inB):
    num=float(inA.T*inB)
    denom=la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)
#2.基于物品相似度额推荐引擎
#用来计算在给定相似度计算方法的条件下，用户对物品的估计评分值
def standEst(dataMat, user, simMeas, item):#数据矩阵、用户编号、相似度计算方法、物品编号，其中行对应用户，列对应物品
    n = shape(dataMat)[1]#得到物品的数目
    simTotal = 0.0; ratSimTotal = 0.0#用于计算估分值的变量初始化
    for j in range(n):#遍历每一个物品
        userRating = dataMat[user,j]#获取用户评分
        if userRating == 0: continue#如果物品评分为0，说明用户未对该物品评分，跳过
        #给出两个物品当中已经被评分的那个元素
        overLap = nonzero(logical_and(dataMat[:,item].A>0, dataMat[:,j].A>0))[0]
        #如果两者没有任何重合元素，则相似度为0，终止此次循环
        if len(overLap) == 0: similarity = 0
        #如果存在重合物品，则基于这些重合物品计算相似度
        else: similarity = simMeas(dataMat[overLap,item],  dataMat[overLap,j])
        print ('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal
    
#基于SVD的评分估计
def svdEst(dataMat,user,simMeas,item):
    n=shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    #对数据集进行SVD分解，分解后只利用90%的奇异值
    U,Sigma,VT=la.svd(dataMat)
    #建立对角矩阵
    Sig4=mat(eye(4)*Sigma[:4])
    #利用U矩阵将物品转换到低维空间中
    xformedItems=dataMat.T*U[:,:4]*Sig4.I
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating==0 or j==item:continue
        similarity = simMeas(xformedItems[item,:].T,xformedItems[j,:].T)
        print ('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal

#推荐引擎 ,产生最高的N个推荐结果
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    #对给定用户建立一个未评分的物品列表
    unratedItems = nonzero(dataMat[user,:].A==0)[1]
    if len(unratedItems) == 0: return ('you rated everything')
    itemScores = []
    for item in unratedItems:#对每个未评分物品
        #调用上述方法产生预测得分
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]
  
#应用实例：利用svd应用于图像压缩
#打印矩阵
def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                print (1),
            else: print (0),
        print ('')
#实现图像的压缩
def imgCompress(numSV=3, thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print ('****original matrix******')
    printMat(myMat, thresh)
    U,Sigma,VT = la.svd(myMat)
    SigRecon = mat(zeros((numSV, numSV)))
    for k in range(numSV):#construct diagonal matrix from vector
        SigRecon[k,k] = Sigma[k]
    reconMat = U[:,:numSV]*SigRecon*VT[:numSV,:]
    print ('****reconstructed matrix using %d singular values******" % numSV')
    printMat(reconMat, thresh)
    
      
  
'''    
myMat=mat(loadExData())
myMat[0,1]=myMat[0,0]=myMat[1,0]=myMat[2,0]=4
myMat[3,3]=2  
print(myMat)
print(recommend(myMat, 2))
'''
'''
myMat=mat(loadExData2())
print(recommend(myMat, 1,estMethod=svdEst))
'''
imgCompress(3)