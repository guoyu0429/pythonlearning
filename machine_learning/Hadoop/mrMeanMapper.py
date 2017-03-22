#coding=utf-8
#分布式均值和方差的mapper
'''
mapper:单个作业被分成很多小粉，输入数据也被切片分发到每个节点，各个节点只在本地数据上做运算，对应的运算代码成为mapper
，这个过程成为map阶段,每个mapper的输出通过某种方式组合，排序后的结果再被分成小份分发到各个节点进行下一步处理工作。
'''

import sys
from numpy import mat,mean,power

def read_input(file):
    for line in file:
        yield line.rstrip()
        
input=read_input(sys.stdin)#按行读取数据
input=[float(line) for line in input]#创建一组对应的浮点数
numInputs=len(input)#得到数组的长度
input=mat(input)#创建numpy矩阵
sqInput=power(input,2)#对所有值平方

print('%d\t%f\t%f' %(numInputs,mean(input),mean(sqInput)))#将均值和平方后的均值输出
print(sys.stderr,'report:still alive')