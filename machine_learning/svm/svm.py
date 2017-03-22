#coding=utf-8
#svm
from numpy import *
#1.打开文件(辅助函数）
def loadDataSet(fileName):
    dataMat=[]
    labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat
#2.i是alpha的下标，m是所有alpha的数目  生成一个随机数（辅助函数）
def selectJrand(i,m):
    j=i
    while(j==i):
        j=int(random.uniform(0,m))
    return j
#3.阈值函数（辅助函数）
def clipAlpha(aj,H,L):
    '''
    根据公式alpha[j]范围，重新求取alpha[j]，公式如下：  
                #如果 alpha[j]>H      那么alpha[j]=H  
                #如果 L<=alpha[j]<=H  那么不需要更新  
                #如果 alpha[j]<L      那么alpha[j]=L 
    '''
    if aj>H:
        aj=H
    if L<aj:
        aj=L 
    return aj

#4.简化版SMO算法
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):#参数：数据集，类别标签、常数C，容错率、取消前最大的循环次数
    #参数初始化
    dataMatrix=mat(dataMatIn)#将输入数据转换成numoy矩阵，简化操作
    labelMat=mat(classLabels).transpose()#将类别标签转换成矩阵并转置为一个列向量
    b=0
    m,n=shape(dataMatrix)#得到矩阵的行数和列数
    alphas=mat(zeros((m,1)))#初始化alphas矩阵
    iter=0#在没有任何alpha改变的情况下遍历数据集的次数
    while (iter<maxIter):
        alphaPairsChanged=0#用户记录alpha是否已经进行优化
        for i in range(m):#对集合进行遍历
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b#预测的类别
            Ei=fXi-float(labelMat[i])#预测的和实际的误差
            '''
            #满足KKT条件：  
            #1.label[i]*fxi>1  &&  alpa[i]==0  
            #2.label[i]*fxi==1 &&  0<alpa[i]<C  
            #3.label[i]*fxi<1  &&  alpa[i]=C  
            #根据定义的EI，可知根据符号 EI*label[i]与零比较，等价于上面的KKT条件  
            #那么不满足KKT条件的为：  
            #1、EI*label[i]>0  &&   alpa[i]>0    需要做优化  
            #2、EI*label[i]==0 &&   这个时候数据点i位于边界上，不做优化处理  
            #3、EI*label[i]<0  &&   alpa[i]<C    需要做优化  
            '''
            if((labelMat[i]*Ei<-toler) and (alphas[i]<C)) or ((labelMat[i]*Ei>toler) and (alphas[i]>0)):#如果误差很大进行优化
                j=selectJrand(i, m)#随机选择第二个alpha值
                fXj=float(multiply(alphas, labelMat).T*dataMatrix*dataMatrix[j,:].T)+b#预测类别
                Ej=fXj-float(labelMat[j])#预测误差
                alphaIold=alphas[i].copy()#拷贝alpha的原先的值
                alphaJold=alphas[j].copy()
                if(labelMat[i]!=labelMat[j]):#将alpha[j]的值调整到0到C
                    L=max(0,alphas[j]-alphas[i])
                    H=min(C,C+alphas[j]-alphas[i])
                else:
                    L=max(0,alphas[j]+alphas[i]-C)
                    H=min(C,alphas[j]+alphas[i])
                if L==H:
                    print('L==H')
                    continue
                #eta是alpha[j]的最优修改量
                eta=2.0*dataMatrix[i,:]*dataMatrix[j,:].T-dataMatrix[i,:]*dataMatrix[i,:].T-dataMatrix[j,:]*dataMatrix[j,:].T
                if eta>=0:
                    print('eta>=0')
                    continue
                alphas[j]-=labelMat[j]*(Ei-Ej)/eta
                alphas[j]=clipAlpha(alphas[j], H, L)
                #检查alpha[j]是否有轻微的改变
                if(abs(alphas[j]-alphaJold)<0.00001):
                    print('j not moving enough')
                    continue
                #对alpha[i]做同样的改变（改变的方向正好相反，即一个增大，一个减小）
                alphas[i]+=labelMat[j]*labelMat[i]*(alphaJold-alphas[j])
                #更新参数b 分别根据公式 计算b1、b2 并计算b值 
                b1=b-Ei-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T   
                b2=b-Ej-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j, :].T
                if(0<alphas[i]) and (C>alphas[i]):
                    b=b1
                elif(0<alphas[j]) and (C>alphas[j]):
                    b=b2
                else:
                    b=(b1+b2)/2.0
                alphaPairsChanged+=1
                print ('iter: %d i: %d, pairs changed %d' %(iter, i, alphaPairsChanged) )
        if(alphaPairsChanged==0):   
            iter+=1   
        else:   #alpha有更新，将iter设为0继续进行程序
            iter=0  
        print('iteration number:%d' %iter)
    return b,alphas     

#核函数：将低维空间映射到高维空间
def kernelTrans(X, A, kTup): #calc the kernel or transform data to a higher dimensional space
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0]=='lin': K = X * A.T   #linear kernel
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2)) #divide in NumPy is element-wise not matrix like Matlab
    else: raise NameError('Houston We Have a Problem That Kernel is not recognized')
    return K
    
class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters 
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) #first column is valid flag
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)


#利用核函数进行分类的径向基测试函数
def testRbf(k1=1.3):
    dataArr,labelArr=loadDataSet('testSetRBF.txt')
    b,alphas=smoP(dataArr,200,0.0001,10000,('rbf',k1))
    datMat=mat(dataArr);labelMat=mat(labelArr).transpose()
    #取得支持向量的索引
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd]
    labelSV=labelMat[svInd]
    print('there are %d support vectors' %shape(sVs)[0])
    m,n=shape(datMat)
    errorCount=0
    for i in range(m):
        kernelEvl=kernelTrans(sVs,datMat[i,:],('rbf',k1))
        #计算输出公式
        predict=kernelEvl.T*multiply(labelSV,alphas[svInd])+b
        if sign(predict)!=sign(labelArr[i]):
             errorCount+=1.0
    print('the training error rate is:%f' %(float(errorCount)/m))
    dataArr,labelArr=loadDataSet('testSetRBF2.txt')
    errorCount=0
    for i in range(m):
        kernelEval=kernelTrans(sVs,datMat[i,:],('rbf',k1));
        predict=kernelEval.T*multiply(labelSV,alphas[svInd])+b;
        if sign(predict)!=sign(labelArr[i]):
            errorCount+=1.0;
    print('the test error rate is:%f '%(float(errorCount)/m));     
    

#误差缓存       
def calcEk(oS,k):
    fXk=float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k]+oS.b)
    Ek=fXk-float(oS.labelMat[k])
    return Ek
#用于选择第二个alpha的值
def selectJ(i,oS,Ei):
    maxK=-1
    maxDeltaE=0
    Ej=0
    #将第i个数据算出来的误差存入数组中，1代表数据有效
    oS.eCache[i]=[1,Ei]
    #A运算表示，将矩阵转换成array
    validEcacheList=nonzero(oS.eCache[:0].A)[0]#构建一个非零表
    if(len(validEcacheList))>1:
        #根据启发式的原理，第二个数据点的选择是选取具有最大|E1-E2|的数据点，因为迭代步长正比于|E1-E2|
        for k in validEcacheList:
            if k==i:continue
            Ek=calcEk(oS, k)
            deltaE=abs(Ei-Ek)
            if(deltaE>maxDeltaE):
                maxK=k 
                maxDeltaE=deltaE
                Ej=Ek
        return maxK,Ej
    #选择具有最大步长的j
    else:
        j=selectJrand(i, oS.m)
        Ej=calcEk(oS, j)
    return j,Ej

#计算误差值并存入缓存当中
def updateEk(oS,k):
    Ek=calcEk(oS, k)
    oS.eCache[k]=[1,Ek]

#根据第一个点，调用上面的方法选择第二个点，并对系数进行更新，返回是否更新了数据对      
def innerL(i,oS):
    Ei=calcEk(oS, i)
    if((oS.labelMat[i]*Ei<-oS.tol) and (oS.alphas[i]<oS.C)) or ((oS.labelMat[i]*Ei>oS.tol) and (oS.alphas[i]>0)):
        j,Ej=selectJ(i, oS, Ei)
        alphaIold=oS.alphas[i].copy()
        alphaJold=oS.alphas[j].copy()
        if(oS.labelMat[i]!=oS.labelMat[j]):
            L=max(0,oS.alphas[j]-oS.alphas[i])
            H=min(oS.C,oS.C+oS.alphas[j]-oS.alphas[i])
        else:
            L=max(0,oS.alphas[j]+oS.alphas[i]-oS.C)
            H=min(oS.C,oS.alphas[j]+oS.alphas[i])
        if L==H:
            print('L==H');return 0
        eta=2.0*oS.K[i,j]-oS.K[i,i]-oS.K[j,j]
        if eta>=0:
            print('eta>=0');return 0
        oS.alphas[i]-=oS.labelMat[j]*(Ei-Ej)/eta
        oS.alphas[j]=clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if(abs(oS.alphas[j]-alphaJold)<0.00001):
            print('j not moving enough');return 0
        oS.alphas[i]+=oS.labelMat[j]*oS.labelMat[i]*(alphaJold-oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if(0<oS.alphas[i]) and (oS.C>oS.alphas[i]):
           oS.b=b1
        elif(0<oS.alphas[j]) and (oS.C>oS.alphas[j]):
            oS.b=b2
        else:
            oS.b=(b1+b2)/2.0
        return 1
    else:return 0
 
#完整版的plattSMO算法           
def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter<maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):        
                alphaPairsChanged += innerL(i,oS)
                print ('fullSet, iter: %d i:%d, pairs changed %d' % (iter,i,alphaPairsChanged))
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print ('non-bound, iter: %d i:%d, pairs changed %d' % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True  
        print ('iteration number: %d' % iter)
    return oS.b,oS.alphas


#计算超平面w
def calcWs(alphas,dataArr,classLabels):
    X=mat(dataArr)
    labelMat=mat(classLabels).transpose()
    m,n=shape(X)
    w=zeros((n,1))
    for i in range(m):
        w+=multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w
 
    
'''         
dataArr, labelArr=loadDataSet('testSet.txt')   
b,alphas=smoSimple(dataArr, labelArr, 0.6, 0.001, 40)     
#print (b,alphas)

shape(alphas[alphas>0])
for i in range(100):
    if alphas[i]>0.0:
        print(dataArr[i],labelArr[i])
'''
'''
dataArr, labelArr=loadDataSet('testSet.txt')  
b,alphas=smoP(dataArr, labelArr, 0.6, 0.001, 40)
ws=calcWs(alphas, dataArr, labelArr)
#对某一个数据进行分类
dataMat=mat(dataArr)
print(dataMat[0]*mat(ws)+b)#wx=b,若该值大于0，则属于1类
print(labelArr[0])
print(ws)
'''

#testRbf()