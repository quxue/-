
# coding: utf-8

# In[15]:


import matplotlib.pyplot as  plt
import numpy as np
import operator
#将文件格式的数据转为可操作的特征矩阵和标签
def file2matrix(filename):
    fp = open(filename)
    arrayOLines = fp.readlines()
    numberOfLine = len(arrayOLines)
    returnMat = np.zeros((numberOfLine,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line =line.strip()
        listFromLine =line.split("\t")
        returnMat[index,:]=listFromLine[0:3]
        if listFromLine[-1] == "didntLike":
            classLabelVector.append(1)
        elif listFromLine[-1] == "smallDoses":
            classLabelVector.append(2)
        elif listFromLine[-1] == "largeDoses":
            classLabelVector.append(3)
        index += 1
    return returnMat,classLabelVector
#数据归一化
def autoNorm(returnMat):
    minValues = returnMat.min(0)
    maxValues = returnMat.max(0)
    ranges = maxValues-minValues
    normDateMat = np.zeros(np.shape(returnMat))
    m = normDateMat.shape[0]
    normDateMat = returnMat - np.tile(minValues,(m,1))
    normDateMat = normDateMat/ np.tile(ranges,(m,1))
    return normDateMat,ranges,minValues
#数据可视化
def showDate(normDateMat,classLabelVector):
    fig=plt.figure
    ax=plt.subplot(111)
    ax.scatter(normDateMat[:,1],normDateMat[:,2],15.0*np.array(classLabelVector),np.array(classLabelVector))
    plt.xlabel("fly distance")
    plt.ylabel("number of ice-cream/")
    plt.show()
#将归一化后的图像与原来的图像放到同一个画布坐标轴上
'''
def MshowDate(normDateMat,returnMat,classLabelVector):
  #  fig=plt.figure()
    fig,ax=plt.subplots(ncols=2,nrows=1,sharex=True,sharey=True)
    ax[0].scatter(returnMat[:,1],returnMat[:,2],15.0*np.array(classLabelVector),np.array(classLabelVector))
    ax[1].scatter(normDateMat[:,1],normDateMat[:,2],15.0*np.array(classLabelVector),np.array(classLabelVector))
    plt.xlabel("fly distance")
    plt.ylabel("number of ice-cream")
    plt.show()
'''
def classify0(inX,classLabelVector,normDateMat,k):
    #求出测试数据和训练集的距离
    dateSize = normDateMat.shape[0]
    diffMat = np.tile(inX,(dateSize,1))-normDateMat
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis =1)
    distance = sqDistance ** 0.5
    #选出前k个距离最近的数据和其对应的类别
    sortedDistance = distance.argsort()
    classCount ={}
    for i in range(k):
        voteLabel = classLabelVector[sortedDistance[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0)+1
    #选出这k个值里面次数多，即值最大的那个，即按值对字典排序
    classCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse = False)
    return classCount[0][0]
def classifyPerson():
    resultList=["讨厌","有些喜欢","很喜欢"]
    percentTats = float(input("the time of watching TV"))
    ffMiles = float(input("the miles of flying"))
    iceCream = float(input("the number of eating ice-cream"))
    filename = "F:\python_daima\dating\\datingTestSet.txt"
    #生成训练集
    datingMat,datingLabel = file2matrix(filename)
    normDatingMat,ranges,minValues = autoNorm(datingMat)
    #生成测试集
    inArr = np.array([percentTats,ffMiles,iceCream])
    normInArr = (inArr-minValues)/ranges
    classfierResult = classify0(normInArr,datingLabel,normDatingMat,3)
    print("You may %s this person" % resultList[classfierResult])
if __name__ == "__main__":
    classifyPerson()
   
    

