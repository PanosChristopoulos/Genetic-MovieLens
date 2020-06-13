import pandas as pd


movie_columns = ['movie_id','movie_name','date','url']
movies = pd.read_csv('ml-100k/u.item',sep='|', names=movie_columns, encoding='latin-1')
print(movies)