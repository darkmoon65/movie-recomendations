import pandas as pd
# import movie data set and look at columns
movie = pd.read_csv("movie.csv")
movie = movie.loc[:,["movieId","title"]]

rating = pd.read_csv("rating.csv")
rating = rating.loc[:,["userId","movieId","rating"]]

data = pd.merge(movie,rating)
data = data.head(100000)
data.to_csv('datos.csv')
