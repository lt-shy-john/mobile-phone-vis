
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import *

def normalizeList(L, minValue=0.0, maxValue=1.0):
    L = L - min(L)
    L = L / (max(L) - min(L))
    return L * (maxValue - minValue) + minValue

#Read data file from excel spreadsheet
dataFileName = 'Mobile Device Data for Assignment 2.xlsx'
dataFrame = pd.read_excel(dataFileName)
dataArray = dataFrame.to_numpy().T
attributeArray = dataArray[4:]
attributeNames = dataFrame.columns.values[4:]
releaseYear = dataArray[2].astype(float)
modelNames = dataFrame['Model'].astype(str)
#useLogScale = [True, True, True, False, False, False, False, False, False, False, False, False]

#Data to be used in scatter plots
scatterDatNames = np.array(['RAM', 'Storage', 'CPU', 'Diplay Diagonal', 'Pixel Density'])
useLogScale = np.array([True, True, True, False, False])
scatterData = np.stack((attributeArray[0], attributeArray[1], attributeArray[2], attributeArray[3], attributeArray[11]), axis=1).T

#Get data about phone companies
deviceAndCompanySheet = pd.read_csv('USE THIS DATASET!!! Mobile Device Data Aligned with Company Name and ID.csv')
companyNames = np.array(deviceAndCompanySheet['Company_real']).astype(str)
usedNames, count = [], 0
for i, name in enumerate(companyNames):
    if name in usedNames:
        j = usedNames.index(name)
    else:
        usedNames.append(name)
        count += 1

def getEarliestAdaptersIDs(att, useLogScale):
    att = att.astype(float)
    if useLogScale:
        att[att==0] = np.min(att[att!=0])
        att = normalizeList(np.log(att))
    
    #Perform clustering algorithm in one dimension
    clusters = DBSCAN(eps=0.02).fit(att.reshape(-1, 1))
    labels = clusters.labels_
    
    #Find the ID of the earliest adapters
    firstInClusterIDs = []
    arrayIndex = np.arange(len(att))
    for idx in np.unique(labels):
        isInCluster = (labels == idx)
        firstInClusterID = np.min(arrayIndex[isInCluster])
        firstInClusterIDs.append(firstInClusterID)
    return np.array(firstInClusterIDs)

#Create a list of the index of each earliest adapter for each attribute
def getEarlyAdapterIds():
    earlyAdaptersIDs = []
    for i, dat in enumerate(scatterData):
        for earlyAdap in getEarliestAdaptersIDs(dat, useLogScale[i]):
            earlyAdaptersIDs.append(earlyAdap)
    return np.array(earlyAdaptersIDs)

#For each earliest adapter, give it's company a score
def getCompanyNameAndScore():
    compNames, companyScores = [], []
    for idx in getEarlyAdapterIds():
        compName = companyNames[idx]
        if compName in compNames:
            j = compNames.index(compName)
            companyScores[j] += 1
        else:
            compNames.append(compName)
            companyScores.append(1)

    sortedZip = sorted(zip(companyScores, compNames), reverse=True)
    companyNamesSorted = [x for y, x in sortedZip]
    companyScoresSorted = [y for y, x in sortedZip]
    return np.array(companyNamesSorted), np.array(companyScoresSorted)

def getEarlyAdapMatrix():
    earlyAdapters = [getEarliestAdaptersIDs(dat, useLogScale[i]) for i, dat in enumerate(scatterData)]
    compNames = np.unique(companyNames)
    dataMatrix = np.zeros((len(compNames), len(scatterDatNames)))
    
    for i, dat in enumerate(earlyAdapters):
        for d in dat:
            j = list(compNames).index(companyNames[d])
            dataMatrix[j, i] += 1
    boolArray = np.sum(dataMatrix, axis=1) != 0
    return dataMatrix[boolArray], compNames[boolArray]
        

if __name__ == '__main__':
    getEarlyAdapMatrix()
