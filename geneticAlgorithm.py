import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn import preprocessing
from scipy.stats import pearsonr
import random
from loadMovies import *
from crossover import *
import matplotlib.pyplot as plt

data_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
data = pd.read_csv("ml-100k/u.data", sep='\t', names=data_columns, encoding='UTF-8')


moviesUserTable1=pd.pivot_table(data,values='rating',index='user_id',columns='movie_id')
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

#Δημιουργία one hot encoded πινάκων
oneHotEncodedRatings = ratingsPreprocessed.assign(finalRating = 1)
oneHotEncodedTable = pd.pivot_table(oneHotEncodedRatings,values='finalRating',index='user_id',columns='movie_id').fillna(0)


#Δημιουργία ενός one hot πίνακα με βάρη, έτσι ώστε μία καταχωρημένη τιμή να έχει μεγαλύτερη πιθανότητα επιλογής από μία που δεν υπάρχει
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
    selectedMovieNames = []
    for x in selectedMovies:
        selectedMovieNames.append(movieNameById(x))
    #print("Selected Movie names:")
    #for elem in selectedMovieNames:
    #    print(elem) 
    return [selectedMovies,initialPopulationGenes]


#Συνάρτηση για εύρεση αρχικού πληθυσμού τυχόντων γειτόνων του χρήστη της παραπάνω συνάρτησης με βάση τις επιλεγμένες ταινίες
def userPopulationBySelectedMovies(userID,selectedMovies):
    initPop = initialPopulationFull(userID)
    initialPopulationGenesNeighbor = []
    for y in range(len(selectedMovies)):
        initialPopulationGenesNeighbor.append(initPop[y])
    return initialPopulationGenesNeighbor

#Συνάρτηση Εύρεσης Καταλληλότερου Γονέα με βάση τα αρχικά χρωμοσώματα και το υπάρχον mating Pool
def fitnessFunction(initialPopulationChromosomes,matingPool,userMean):
    pearsonSimilarityList = []
    fitnessSum = 0
    fitnessAvg = 0
    for x in range(len(matingPool)):
        pearsonSimilarityList.append(pearsonr(initialPopulationChromosomes,matingPool[x])[0])
    fittestParentChr = pearsonSimilarityList.index(max(pearsonSimilarityList)) 
    for y in range(len(pearsonSimilarityList)):
        fitnessSum = fitnessSum + pearsonSimilarityList[y]
        fitnessAvg = (fitnessSum/len(pearsonSimilarityList) + userMean)/2
    return [fittestParentChr,fitnessAvg]

def elitism(initialPopulationChromosomes,matingPool,userMean):
    return fitnessFunction(initialPopulationChromosomes,matingPool,userMean)


#Γενετικός Αλγόριθμος με είσοδο το userID
def geneticAlgorithm(userID):
    #Αρχικοποίηση πληθυσμού του χρήστη και εύρεση 10 πιο κοντινών γειτόνων
    initPop = initialPopulationFull(userID)
    tenSimilar = pearsonbyuser(userID)
    print("User",userID,"'s most similar users are:",tenSimilar)

    #Είσοδος του μεγέθους αρχικού πληθυσμού και επιλογή αρχικού πληθυσμού με τη μέθοδο ρουλέτας με βάση το κόστος
    size = int(input("Size of Initial Population: "))
    initialPopulationUser = initialPopulation(userID,size)
    selectedMoviesbyUser = initialPopulationUser[0]
    initialPopulationChromosomes = initialPopulationUser[1]
    userSum = 0
    for x in range(len(initialPopulationUser[1])):
        userSum = userSum + initialPopulationUser[1][x]
    userMean = userSum/len(initialPopulationUser[1])
    #Εμφάνιση επιλεγμένων ταινιών και αρχικών χρωμοσωμάτων
    if size<50:
        try:
            selectedMovieNames = []
            for x in selectedMoviesbyUser:
                selectedMovieNames.append(movieNameById(x))
            print("User no:",userID,"selected movies as chromosomes names are",selectedMovieNames)
        except:
            print("Can't display movie names")
    #print("User no:",userID,"initial population chromosomes are",initialPopulationChromosomes)
    #Ορισμός Mating Pool
    matingPool = []
    for x in range(len(tenSimilar)):
        matingPool.append(userPopulationBySelectedMovies(tenSimilar[x],selectedMoviesbyUser))
    #print(matingPool)

    generationNum =  int(input("Number of Generations "))
    #Εύρεση Καταλληλότερου συζυγούς γονεά για τον χρήστη userID
    fittestParentGA = fitnessFunction(initialPopulationChromosomes,matingPool,userMean)
    generationList = []
    generationFittness = []
    fitnessParentOverallList = []
    for x in range(generationNum):
        #Χρησιμοποιείται ελιτισμός, αφού με κάθε νέα γενιά, αντικαθίσταται το χρωμόσωμα με τον καλύτερο πιθανό γονέα από το mating pool
        effectCoeff = random.randint(1, 100)
        elitismCoeff = 0
        crossoverCoeff = 40
        fittestParentGA1 = fitnessFunction(initialPopulationChromosomes,matingPool,userMean)
        crossover(initialPopulationChromosomes,matingPool[fittestParentGA1[0]])
        generationList.append(x)
        generationFittness.append(fittestParentGA1[1])
        print("Generation: ",x+1,"Generation Fitness",fittestParentGA1[1])
        fitnessParentOverallList.append(tenSimilar[fittestParentGA1[0]])
    uniquefitnessParentOverallList = []  
    for x in fitnessParentOverallList: 
        if x not in uniquefitnessParentOverallList: 
            uniquefitnessParentOverallList.append(x)
    for x in range(len(uniquefitnessParentOverallList)):
        findMoviesToPropose(userID,uniquefitnessParentOverallList[x])
    #plt.plot(generationList,generationFittness)
    #plt.show()

geneticAlgorithm(70)


"""

    for x in range(len(uniquefitnessParentOverallList)):
        a = oneHotEncodedTable.loc[userID].values
        b = oneHotEncodedTable.loc[uniquefitnessParentOverallList[x]].values
        fitParList = moviesUserTable1.loc[uniquefitnessParentOverallList[x]].tolist()
        moviesNotRated = []
        for x in range(len(a)):
            if b[x] == 1 and a[x] == 0:
                moviesNotRated.append(x)
        moviesToPropose = []
        for x in range(len(moviesNotRated)):
            if fitParList[x] > 2:
                moviesToPropose.append(x)
        print(moviesToPropose)
        for x in range(len(moviesToPropose)):
            try:
                printMovieAndRatingByUser(uniquefitnessParentOverallList[x],moviesToPropose[x]+1)
            except:
                print(" ")

"""