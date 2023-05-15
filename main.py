
# Importing the required libraries
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import pickle
import warnings

warnings.filterwarnings('ignore')

# Importing the csv file data
movies = pd.read_csv('data/tmdb_5000_movies.csv')
credit = pd.read_csv('data/tmdb_5000_credits.csv')

# Displaying the top 5 rows of data for both dataframes
print("\n Top 5 rows of Movies DataFrame: \n")
print(movies.head())

print("\n Top 5 rows of Credits DataFrame: \n")
print(credit.head())

# Displaying the shape of data for both dataframes
print("\n Shape of Movies DataFrame: \n")
print(movies.shape)

print("\n Shape of Credits DataFrame: \n")
print(credit.shape)

# Merging the two dataframes
movies = movies.merge(credit, on='title')

# Keeping only the required columns for our recommendation system
# Genre, Id, Keywords, Title, Overviews, Cast, Director
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Data preprocessing and creating a new dataframe with tags column

# Checking for null values
print("\n Null Values : \n")
print(movies.isnull().sum())

# Dropping null values
movies.dropna(inplace=True)

# Checking for duplicated rows
print("\n Duplicated Rows : \n")
print(movies.duplicated().sum())


# Cleaning the genre,keywords column
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)


# Cleaning the cast column
def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L


movies['cast'] = movies['cast'].apply(convert3)


# Cleaning the crew column

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


movies['crew'] = movies['crew'].apply(fetch_director)

# Converting the 'overview' column to list
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Removing the spaces
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

# Creating a new tags column
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Creating a new dataframe with the required columns
new_df = movies[['movie_id', 'title', 'tags']]

# Convert the list into string of tag column
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# Convert the tags column to lower case
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# Text processing with NLP


# Stemming
ps = PorterStemmer()


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


new_df['tags'] = new_df['tags'].apply(stem)

# Removing stop words
# Text vectorisation using bag of words

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# Content Based Recommendation System

# Finding cosine similarity

similarity = cosine_similarity(vectors)


# Making a recommendation function

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# Pickle codes
pickle.dump(new_df.to_dict(), open('movie_dict.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

