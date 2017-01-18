from numpy import *

def loadDataet():
    postingList = [['my','dog','has','flea',\
                    'problems','help','please'],
                    ['maybe','not','take','him',\
                    'to','dog','park','stupid'],
                    ['my','dalmation','is','so','cute',\
                    'I','love','hime'],
                    ['stop','posting','stupid','worthless','garbage'],
                    ['mr','licks','ate','my','steak','how',\
                    'to','stop','him'],
                    ['quit','buying','worthless','dog','food','stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList, classVec

def createVocabList(dataSet):
    vocaSet = set([])
    for document in dataSet:
        vocaSet = vocaSet | set(document)
    return list(vocaSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("The word is not int my Vocabulary")
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = zeros(numWords)
    p1Num = zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)  # change to log()
    p0Vect = log(p0Num / p0Denom)  # change to log()
    return p0Vect, p1Vect, pAbusive

listOPosts, listClasses = loadDataet()
myVocabList = createVocabList(listOPosts)
trainMat = []
for postinDoc in  listOPosts:
    trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
p0V,p1V,pAb=trainNB0(trainMat,listClasses)