import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn import preprocessing
from scipy.stats import pearsonr

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
oneHotRatings = ratingsPreprocessed.assign(finalRating = 1)
oneHotTable = pd.pivot_table(oneHotRatings,values='finalRating',index='user_id',columns='movie_id').fillna(0)





#Συνάρτηση για την εύρεση του Pearson Similarity ανάμεσα σε χρήστες
def pearsonSimilarity(user1,user2):
	user1row=finalMoviesUserTable.loc[user1].values
	user2row=finalMoviesUserTable.loc[user2].values
	k =pearsonr(user1row,user2row)[0]
	return float("{:.3f}".format(k))


similarityDict = {}
def pearsonbyuser(user1):
    for x in range(1,943):
        similarityDict[x]=pearsonSimilarity(user1,x)
    dict2 = {k: v for k, v in sorted(similarityDict.items(), key=lambda item: item[1])}
    dict2 =  {key:val for key, val in dict2.items() if val == val}
    if user1 in dict2: del dict2[user1]
    tenSimilar = []
    for x in range(1,11):
        tenSimilar.append(list(dict2)[-x])
    return tenSimilar

