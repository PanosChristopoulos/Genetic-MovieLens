import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn import preprocessing
from scipy.stats import pearsonr
import random
from loadMovies import *

  
#Συνάρτηση υλοποίησης crossover μοναδικού σημείου
def crossover(chr1, chr2): 

    #Τυχαίος αριθμός τίθεται ως crossover point
    randomCrossoverPoint = random.randint(0, len(chr1)) 
    print("Crossover point :", randomCrossoverPoint,"out of",len(chr1)," -> ",chr1[:randomCrossoverPoint], " + ", chr2[randomCrossoverPoint:]) 
  
#Διασταύρωση γονιδίων
    for i in range(randomCrossoverPoint, len(chr2)):
        eris1 = chr1
        eris2 = chr2
        chr1[i] = eris2[i]
        chr2[i] = eris1[i]
    return chr1, chr2 


