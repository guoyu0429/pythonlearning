#coding=utf-8
#分布式计算均值和方差的reducer
'''
第二部的处理阶段被称为reduce阶段，对应的运行代码成为reducer，reduce的输出就是程序的最终执行结果
'''
import sys 
from numpy import mat,mean,power

def read_input(file):
    for line in file:
        yield line.rstrip()
        
input=read_input(sys.stdin)
mapperOut=[line.split('\t') for line in input]
cumVal=0.0
cumSumSq=0.0
cumN=0.0
for instance in mapperOut:
    nj=float(instance[0])
    cumN+=nj
    cumVal+=nj*float(instance[1])
    cumSumSq+=nj*float(instance[2])
mean=cumVal/cumN
varSum=(cumSumSq-2*mean*cumVal+cumN*mean*mean)/cumN
print('%d\t%f\t%f' %(cumN,mean,varSum))
print(sys.stderr,'report:still alive')
