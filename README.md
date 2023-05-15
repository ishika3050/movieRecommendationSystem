# Movie Recommendation System

Movie Recommendation System, a project developed in Python using a content-based recommendation algorithm to suggest the top 5 movies to watch based on user-selected movies.

## Description
- This is a movie recommendation system built using Python and its libraries, which has been deployed using Streamlit. 
- The dataset for the system has been sourced from Kaggle, and it employs a content-based recommendation algorithm. 
- The user selects a movie from a list of available options, and the system suggests the top 5 movies to watch based on the similarity of their content to the chosen movie.
- The system's user interface has been designed with Streamlit to provide a seamless user experience.

#### Content Based Recommendation System
- A content-based recommendation system is a type of recommendation system that makes recommendations based on the similarity of the items being recommended. 
- It works by analyzing the features or attributes of the items, such as genre, actors, or plot keywords, and then finding other items that share similar attributes. 
- The system then recommends these similar items to the user. 
- Content-based recommendation systems are particularly useful when the user has a clear idea of what they are looking for, such as a specific genre or actor, and can provide personalized recommendations based on the user's preferences. 


## Prerequisites 

### Dataset 
The dataset for this project has been sourced from Kaggle and is based on data from TMDb (The Movie Database) website. 
To use this system, you will need to download the two CSV files from Kaggle - 'tmdb_5000_movies.csv' and 'tmdb_5000_credits.csv' - which contain information about movie titles, ratings, cast, crew, and other relevant details. 
The link to the dataset -
https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata?select=tmdb_5000_movies.csv

### Downloading Requirements

Use the following command to install all the python modules and packages mentioned in the requirements.txt file : 

pip install -r requirements.txt

### Installation and Usage

To run this project, follow these steps:

1. Download the dataset from Kaggle and store it in a folder named "data" in the root directory of the project. The two required CSV files are "tmdb_5000_movies.csv" and "tmdb_5000_credits.csv".
2. Install the required Python libraries by running pip install -r requirements.txt in your terminal.
3. Run the main script by running python main.py in your terminal. This will preprocess the data and generate the necessary pickle files for the app to run.
4. Finally, run the Streamlit app by running streamlit run app.py in your terminal. This will launch the app in your browser, where you can choose a movie from the list and receive personalized recommendations based on a content-based recommendation system.

- Note: Make sure you have Python 3.x installed on your system before running this project.

## Tech 

1. Python
- Python is a programming language that is preferred for programming due to its vast features, applicability, and simplicity. 
- The Python programming language best fits machine learning due to its independent platform and its popularity in the programming community.

2. pandas 
- pandas is a Python package providing fast, flexible, and expressive data structures designed to make working with “relational” or “labeled” data both easy and intuitive. 
- It aims to be the fundamental high-level building block for doing practical, real-world data analysis in Python.

3. NLTK 
- NLTK, or Natural Language Toolkit, is a Python package that you can use for NLP. 
- A lot of the data that you could be analyzing is unstructured data and contains human-readable text. 
- Before you can analyze that data programmatically, you first need to preprocess it.

4. Scikit Learn / Sklearn 
- Scikit-learn (Sklearn) is the most useful and robust library for machine learning in Python. 
- It provides a selection of efficient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction via a consistence interface in Python.

5. Streamlit 
- Streamlit is a Python library that simplifies the process of creating web applications for machine learning and data science projects.

6. Pickle 
- Pickle is a Python module that allows for the serialization and deserialization of Python object structures, enabling easy storage and retrieval of complex data.

7. Cosine similarity 
- Cosine similarity is a measure used to determine the similarity between two vectors in a multi-dimensional space. 
- In content-based recommendation systems, cosine similarity is often used to calculate the similarity between the features or attributes of different items. 
- In recommendation systems, cosine similarity can be used to find items that are most similar to each other, and to calculate how similar a given item is to a user's preferences or previously watched content.

## Author

- [@ishika3050]
(https://github.com/ishika3050)

(ishika3050@gmail.com)

(https://www.linkedin.com/in/ishika3050/)
