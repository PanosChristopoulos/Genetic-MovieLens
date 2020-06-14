import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn import preprocessing
from scipy.stats import pearsonr
import random
from loadMovies import *


data_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
data = pd.read_csv("ml-100k/u.data", sep='\t', names=data_columns, encoding='UTF-8')



#Κεντράρισμα αξιολογήσεων με βάση τη μέση βαθμολογία του χρήστη
meanRating = data.groupby(by="user_id",as_index=False)['rating'].mean()
ratingCentered = pd.merge(data,meanRating,on='user_id')
ratingCentered['finalRating']=ratingCentered['rating_x']-ratingCentered['rating_y']


#Χρήση ενός NumPy array για την απεικόνιση του γενικού συνόλου
numArray = ratingCentered[['finalRating']].values.astype(float)

#Χρήση του MinMax Scaler της βιβλιοθήκης SkLearn και εισαγωγή των νέων τιμών σε ένα
#νέο data frame (ratingNormalized)
min_max_scaler = preprocessing.MinMaxScaler()
numArray_scaled = min_max_scaler.fit_transform(numArray)
ratingNormalized = pd.DataFrame(numArray_scaled)

#Εισαγωγή των νέων κανονικοποιημένων αξιολογήσεων στο συνολικό πίνακα
ratingsPreprocessed = ratingCentered.assign(finalRating=ratingNormalized[0])


#Συμπλήρωση NaN κελιών με μέσο όρο ταινίας
moviesUserTable=pd.pivot_table(ratingsPreprocessed,values='finalRating',index='user_id',columns='movie_id')
finalMoviesUserTable = moviesUserTable.fillna(moviesUserTable.mean(axis=0))




#Δημιουργία ενός oneHot Encoded πίνακα βαθμολογιών
oneHotRatings = ratingsPreprocessed.assign(finalRating = 1.2)
oneHotTable = pd.pivot_table(oneHotRatings,values='finalRating',index='user_id',columns='movie_id').fillna(0.8)




#Συνάρτηση για την εύρεση του Pearson Similarity ανάμεσα σε χρήστες
def pearsonSimilarity(user1,user2):
	user1row=finalMoviesUserTable.loc[user1].values
	user2row=finalMoviesUserTable.loc[user2].values
	k =pearsonr(user1row,user2row)[0]
	return float("{:.3f}".format(k))


#Συνάρτηση για την εύρεση 10 παρόμοιων χρηστών με τον user1
def pearsonbyuser(user1):
    similarityDict = {}
    for x in range(1,943):
        similarityDict[x]=pearsonSimilarity(user1,x)
    dict2 = {k: v for k, v in sorted(similarityDict.items(), key=lambda item: item[1])}
    dict2 =  {key:val for key, val in dict2.items() if val == val}
    if user1 in dict2: del dict2[user1]
    tenSimilar = []
    for x in range(1,11):
        tenSimilar.append(list(dict2)[-x])
    return tenSimilar

#Συνάρτηση για εύρεση αρχικού πληθυσμού σε περίπτωση χρήσης όλων των δεδομένων για κάθε χρήστη
def initialPopulationFull(userID):
    initPop = finalMoviesUserTable.to_numpy()[userID-1]
    #print("User",userID,"'s Full Initial Population is",initPop)
    return initPop


#Συνάρτηση για εύρεση αρχικού πλήθυσμού με χρήστη "num" δεδομένων για τον χρήστη userID - μέθοδος ρουλέτας με βάση το κόστος
def initialPopulation(userID,num):
    initPop = initialPopulationFull(userID)
    selectedMovies = []
    initialPopulationGenes = []
    initOneHotTable = oneHotTable.to_numpy()[userID-1]
    oneHotSum = sum(initOneHotTable)
    #print(oneHotSum)
    rouletteNumbers = []
    for x in range(num):
        num1 = random.randrange(int(oneHotSum))
        rouletteNumbers.append(num1)
        ittNum = 0
        movieCounter = 0
        while ittNum<num1:
            ittNum = ittNum + initOneHotTable[0]
            movieCounter += 1
        selectedMovies.append(movieCounter)
    for y in range(len(selectedMovies)):
        initialPopulationGenes.append(initPop[y])
    print(num,"roulette Numbers", "for", userID, "are:", rouletteNumbers)
    print(num,"selected Movie IDs", "for", userID, "are:", selectedMovies)
    selectedMovieNames = []
    for x in selectedMovies:
        selectedMovieNames.append(movieNameById(x))
    print("Selected Movie names:")
    for elem in selectedMovieNames:
        print(elem) 
    return [selectedMovies,initialPopulationGenes]


#Συνάρτηση για εύρεση αρχικού πληθυσμού τυχόντων γειτόνων του χρήστη της παραπάνω συνάρτησης με βάση τις επιλεγμένες ταινίες
def userPopulationBySelectedMovies(userID,selectedMovies):
    initPop = initialPopulationFull(userID)
    initialPopulationGenesNeighbor = []
    for y in range(len(selectedMovies)):
        initialPopulationGenesNeighbor.append(initPop[y])
    return initialPopulationGenesNeighbor


#Συνάρτηση για επεξεργασία βάση userID
def userProcessing(userID):
    initPop = initialPopulationFull(userID)
    tenSimilar = pearsonbyuser(userID)
    print("User",userID,"'s most similar users are:",tenSimilar)
    size = int(input("Size of Initial Population: "))
    initialPopulationUser = initialPopulation(userID,size)
    selectedMoviesbyUser = initialPopulationUser[0]
    initialPopulationChromosomes = initialPopulationUser[1]
    selectedMovieNames = []
    for x in selectedMoviesbyUser:
        selectedMovieNames.append(movieNameById(x))
    print("User no:",userID,"selected movies are",selectedMovieNames)
    print("User no:",userID,"initial population chromosomes are",initialPopulationChromosomes)
    #for xy in range(len(tenSimilar)):
        #print("user",tenSimilar[xy],"ratings",userPopulationBySelectedMovies(xy,selectedMoviesbyUser))

initialPopulation(61,20)


