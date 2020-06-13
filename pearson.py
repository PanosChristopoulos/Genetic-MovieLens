import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn import preprocessing
from scipy.stats import pearsonr
from geneticAlgorithm import *

#Συνάρτηση για την εύρεση του Pearson Similarity ανάμεσα σε χρήστες
def pearsonSimilarity(user1,user2):
	user1row=final_movie.loc[user1].values
	user2row=final_movie.loc[user2].values
	k =pearsonr(user1row,user2row)[0]
	return float("{:.3f}".format(k))



similarityDict = {}
def pearsonbyuser(user1):
#test = pearsonSimilarity(43,103)
#print(test)
    for x in range(1,943):
        similarityDict[x]=pearsonSimilarity(user1,x)
    dict2 = {k: v for k, v in sorted(similarityDict.items(), key=lambda item: item[1])}
    dict2 =  {key:val for key, val in dict2.items() if val == val}
    if user1 in dict2: del dict2[user1]
    tenSimilar = []
    for x in range(1,11):
        tenSimilar.append(list(dict2)[-x])
    return tenSimilar

