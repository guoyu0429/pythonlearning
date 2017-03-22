import numpy as np
import tkinter as tk
import regTrees

import matplotlib
#设置matplotlib的后端为TkAgg,可在实现在所选GUI框架上调用Agg，把Agg呈现在画布上
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def reDraw(tolS,tolN):
    reDraw.f.clf()        # 清空之前的图像
    reDraw.a = reDraw.f.add_subplot(111)#重新添加一个新图
    if chkBtnVar.get():#检查复选框是否被选中，根据复选框是否被选中来确定构建模型树还是回归树
        if tolN < 2: tolN = 2
        myTree=regTrees.createTree(reDraw.rawDat, regTrees.modelLeaf,regTrees.modelErr, (tolS,tolN))#用真实值构建树
        yHat = regTrees.createForeCast(myTree, reDraw.testDat, regTrees.modelTreeEval)#用测试值构建树
    else:
        myTree=regTrees.createTree(reDraw.rawDat, ops=(tolS,tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.testDat)
    reDraw.rawDat[:,0].A
    reDraw.a.scatter(reDraw.rawDat[:,0], reDraw.rawDat[:,1], s=5) #真实值用scatter()方法绘制，因为scatter构建的是离散型散点图
    reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0) #预测值用plot()方法构建，连续曲线
    reDraw.canvas.show()
    
def getInputs():
    try: tolN = int(tolNentry.get())
    except: 
        tolN = 10 
        print ('enter Integer for tolN')
        tolNentry.delete(0, tk.END)
        tolNentry.insert(0,'10')
    try: tolS = float(tolSentry.get())
    except: 
        tolS = 1.0 
        print ('enter Float for tolS')
        tolSentry.delete(0, tk.END)
        tolSentry.insert(0,'1.0')
    return tolN,tolS

def drawNewTree():
    tolN,tolS = getInputs()#获得输入框的值
    reDraw(tolS,tolN)
    
root=tk.Tk()

reDraw.f = Figure(figsize=(5,4), dpi=100) #create canvas
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

tk.Label(root, text="tolN").grid(row=1, column=0)
tolNentry = tk.Entry(root)
tolNentry.grid(row=1, column=1)
tolNentry.insert(0,'10')
tk.Label(root, text="tolS").grid(row=2, column=0)
tolSentry = tk.Entry(root)
tolSentry.grid(row=2, column=1)
tolSentry.insert(0,'1.0')
#点击redraw按钮调用drawtree函数
tk.Button(root, text="ReDraw", command=drawNewTree).grid(row=1, column=2, rowspan=3)
chkBtnVar = tk.IntVar()
chkBtn = tk.Checkbutton(root, text="Model Tree", variable = chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=2)

reDraw.rawDat = np.mat(regTrees.loadDataSet('sine.txt'))

reDraw.testDat = np.arange(min(reDraw.rawDat[:,0]), max(reDraw.rawDat[:,0]),0.01)  

reDraw(1.0, 10)
               
root.mainloop()



