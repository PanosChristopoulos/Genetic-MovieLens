import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn import preprocessing
from scipy.stats import pearsonr
import random



m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movies = pd.read_csv('ml-100k/u.item', sep='|', names=m_cols, usecols=range(5),
                     encoding='latin-1')

def movieNameById(movieID):
    return movies.loc[movies["movie_id"] == movieID]['title'].item()

data_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
data = pd.read_csv("ml-100k/u.data", sep='\t', names=data_columns, encoding='UTF-8')
moviesUserTable1=pd.pivot_table(data,values='rating',index='user_id',columns='movie_id').fillna(0)
#print(moviesUserTable1)

def printMovieAndRatingByUser(userID,movieID):
    a = moviesUserTable1.loc[userID].tolist()
    print("User ",userID," proposes film", movieNameById(movieID)," as he rated it with", a[movieID-1])


def findMoviesToPropose(user1,user2):
    data1 = moviesUserTable1.loc[user1].tolist()
    data2 = moviesUserTable1.loc[user2].tolist()
    moviesToPropose = []
    for x in range(len(data2)):
        if data2[x]>data1[x]+3:
            moviesToPropose.append(x)
    for x in range(len(moviesToPropose)):
        printMovieAndRatingByUser(user2,moviesToPropose[x]+1)

