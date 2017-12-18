import numpy as np
from scipy.linalg import det, norm, inv, solve, lu, cholesky, eig, svd
import numpy.linalg as nl
import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from random import randrange, uniform
import copy

#example golf data
pg = pd.read_csv('pga_data.csv')
drive = pg["Avg.Drive"]
avgwin = pg["Avg.Win"]
m = np.asfarray(drive)
newdata = np.column_stack((m,np.asfarray(avgwin)))


#feature scale columns for dataframe

def feature_scale(data):
	shape = data.shape
	if(len(shape) > 1):
		for col in range(shape[1]):
			for i in range(shape[0]):
				data[i,col] = float(data[i,col] - np.min(data[:,col]))/float(np.max(data[:,col])-np.min(data[:,col]))
	return data


def calculate_distance(data,clusters,k):
	copclusters = copy.deepcopy(clusters)
	for i in range(k):
		copclusters["cluster" + str(i)]["points"] = []
	for j in range(data.shape[0]):
		newnorms = []
		point = data[j,:]
		for centers in range(k):
			norm = nl.norm((clusters["cluster"+str(centers)]["center"] - point),2)
			newnorms.append(norm)
		index = np.argmin(newnorms)
		copclusters["cluster"+str(index)]["points"].append(point)
	return copclusters



def kmeans(k,data):
	clusters = {}
	for i in range(k):
		clusters["cluster" + str(i)] = {"center":np.asfarray([uniform(np.min(data[:,z]),np.max(data[:,z])) for z in range(data.shape[1])]), "points":[]}
	lclusters = copy.deepcopy(clusters)
	newclusters = calculate_distance(data,lclusters,k)
	count = 0
	boolean = False
	while(boolean == False):
		for o in range(k):
			print len(newclusters["cluster" + str(o)]["points"])
			print "---"
			print len(lclusters["cluster" + str(o)]["points"])
		numcheck = 0
		for check in range(k):
			if(count != 0):
				clstack = np.vstack(lclusters["cluster" + str(check)]["points"])
				nlstack = np.vstack(newclusters["cluster" + str(check)]["points"])
				if np.array_equal(clstack[clstack[:,0].argsort()],nlstack[nlstack[:,0].argsort()]) == True:
					numcheck = numcheck + 1
				nothing = 0
		if numcheck == k:
			return copy.deepcopy(newclusters)
		else:
			boolean = False
		clustercopy = copy.deepcopy(lclusters)
		lclusters = copy.deepcopy(newclusters)
		for j in range(k):
			averages = [0]*data.shape[1]
			if len(newclusters["cluster" + str(j)]["points"]) > 0:
				pointvec = np.vstack(newclusters["cluster" + str(j)]["points"])
				for z in range(data.shape[1]):
					averages[z] = np.mean(pointvec[:,z])
			else:
				averages = clustercopy["cluster" + str(j)]["center"]
			clustercopy["cluster" + str(j)]["center"] = np.asfarray(averages)
		newclusters = calculate_distance(data,clustercopy,k)
		count = count + 1
		print count

	return copy.deepcopy(newclusters)





finalclusters = kmeans(4,newdata)
print finalclusters


















