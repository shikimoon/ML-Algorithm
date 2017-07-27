# -*- coding: UTF-8 -*-
from numpy import *

def loadDataSet():
    data = {}
    data[1] = 5.56
    data[2] = 5.70
    data[3] = 5.91
    data[4] = 6.40
    data[5] = 6.80
    data[6] = 7.05
    data[7] = 8.90
    data[8] = 8.70
    data[9] = 9.00
    data[10] = 9.05
    return data

def getStep(splitList):
    a = min(splitList)-1
    b = max(splitList)+1
    return [i+0.5 for i in range(a,b)]

def splitData(data, stepList):
    error = {}
    splitNum = 0
    splitMean1 = 0
    splitMean2 = 0
    minError = inf
    for i in  stepList:
        mean1 = 0
        mean2 = 0
        a = [v for k, v in data.items() if k < i]
        b = [v for k, v in data.items() if k >= i]
        if len(a) != 0 :
            mean1 = mean(a)
        if len(b) != 0:
            mean2 = mean(b)
        m = [(k-mean1)**2 for k in a]
        n = [(k-mean2)**2 for k in b]
        if sum(m)+sum(n) < minError:
            minError = sum(m)+sum(n)
            splitNum = i
            splitMean1 = mean1
            splitMean2 = mean2
    error["splitNum"] = splitNum
    error["splitMean1"] = splitMean1
    error["splitMean2"] = splitMean2
    error["minError"] = minError
    return error

def addTree(data, data1, stepList):
    error = []
    for i in range(0,7):
        stepError = splitData(data1, stepList)
        updateError(data1,stepError)
        error.append(stepError)
        sumError = calError(data, error)
        print('sumError:',sumError)

def updateError(data, error):
    for k in data.keys():
        if k < error["splitNum"]:
            data[k] -= error["splitMean1"]
        else:
            data[k] -= error["splitMean2"]

def calError(data, error):
    data1 = data.copy()
    sumError = 0
    for k, v in data1.items():
        for eachError in error:
            if k < eachError["splitNum"]:
                v -= eachError["splitMean1"]
            else:
                v -= eachError["splitMean2"]
        sumError += v**2
    return sumError


def test():
    data = loadDataSet()
    data1 = data.copy()
    splitList = [k for k in data.keys()]
    stepList = getStep(splitList)
    addTree(data, data1, stepList)

test()