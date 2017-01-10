# -*- coding: UTF-8 -*-
from numpy import *
import operator
import matplotlib.pyplot as plt



filePath = 'datingTestSet2.txt'

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet-tile(minVals, (m,1))
    normDataSet=normDataSet/tile(ranges, (m,1))
    return normDataSet,ranges,minVals

def classify0(intX,dataSet,labels,k):
    dataSetSize=len(dataSet)
    diffMat=tile(intX, (dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistance=sqDiffMat.sum(axis=1)
    distance=sqDistance**0.5
    sortedDistance=distance.argsort()
    classCount={}
    for i in range(k):
        voteLabel=labels[sortedDistance[i]]
        classCount[voteLabel]=classCount.get(voteLabel,0)+1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def datingClassTest():
    hoRatio=0.10
    datetingDataMat,datingLabels= file2matrix(filePath)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datetingDataMat[:,1], datetingDataMat[:,2], 15.0*array(datingLabels), 15.0*array(datingLabels))
    plt.show()
    normDataSet,ranges,minVals=autoNorm(datetingDataMat)
    m=normDataSet.shape[0]
    numTestVects=int(m*hoRatio)
    print("numTestVects:"+str(numTestVects))
    errorCount=0.0
    for i in range(numTestVects):
        classifierResult=classify0(normDataSet[i,:], normDataSet[numTestVects:m,:], datingLabels[numTestVects:m], 3)
        print("the classifier came back with :%d, the real answer is:%d"%(classifierResult,datingLabels[i]))
        if(classifierResult!=datingLabels[i]):
            errorCount+=1.0
    print("the total error rate is :%f"%(errorCount/float(numTestVects)))

datingClassTest()