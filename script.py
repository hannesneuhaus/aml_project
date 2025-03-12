import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Dict, Text
import tensorflow_recommenders as tfrs
from IPython.display import clear_output


# This is the small dataset
ds_mov = 'Data/ml-latest-small/movies.csv'
ds_rat = 'Data/ml-latest-small/ratings.csv'
ds_tags = 'Data/ml-latest-small/tags.csv'

# This is the entire dataset, because of it's size this will take a while to load
#ds_mov = 'Data/ml-latest/movies.csv'
#ds_rat = 'Data/ml-latest/ratings.csv'
#ds_tags = 'Data/ml-latest/tags.csv'


df_mov = pd.read_csv(ds_mov)
df_rat = pd.read_csv(ds_rat)
df_tags = pd.read_csv(ds_tags)
df_rat = pd.merge(df_rat, df_mov, on="movieId", how="left")
df_mov["movieId"] = df_mov["movieId"].astype(str)
df_rat["userId"] = df_rat["userId"].astype(str)
df_rat["movieId"] = df_rat["movieId"].astype(str)

mov_feat = {key: tf.constant(value) for key, value in df_mov.to_dict(orient="list").items()}
movies = tf.data.Dataset.from_tensor_slices(mov_feat)

rat_feat = {key: tf.constant(value) for key, value in df_rat.to_dict(orient="list").items()}
ratings = tf.data.Dataset.from_tensor_slices(rat_feat)

movies = movies.map(lambda x: x['title'])
ratings = ratings.map(lambda x: {"userId": x["userId"], "title": x["title"], 'rating': x['rating']})

class UserModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_user_ids.astype(str), mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding)
        ])
        
    def call(self, inputs):
        return self.user_embedding(inputs)

class MovieModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.movie_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_movie_titles.astype(str), mask_token=None),
            tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding)
        ])
        
    def call(self, inputs):
        return self.movie_embedding(inputs)

class RecommenderModel(tfrs.Model):
    def __init__(self):
        super().__init__()
        self.user_model = UserModel()
        self.movie_model = MovieModel()
        self.task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
            candidates=movies.batch(128).map(self.movie_model)
        ))
        
    def compute_loss(self, features, training=False):
        print(features)
        user_embeddings = self.user_model(features["userId"])
        movie_embeddings = self.movie_model(features["title"])
        return self.task(user_embeddings, movie_embeddings)

embedding = 64
unique_user_ids = np.unique([x.numpy().decode("utf-8") for x in ratings.map(lambda x: x["userId"])])
unique_movie_titles = np.unique([x.numpy().decode("utf-8") for x in movies])


model = RecommenderModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))


cached_train = ratings.shuffle(100_000).batch(8192).cache()
model.fit(cached_train, epochs=5)

def recommend_movies(model, movie_list, movie_dataset, top_k=3):
    movie_embeddings = model.movie_model(tf.constant(movie_list))
    
    all_movie_titles = [x.numpy().decode("utf-8") for x in movie_dataset]
    all_movie_embeddings = model.movie_model(tf.constant(all_movie_titles))
    
    scores = tf.linalg.matmul(movie_embeddings, all_movie_embeddings, transpose_b=True)
    avg_scores = tf.reduce_mean(scores, axis=0) 
    
    top_indices = tf.argsort(avg_scores, direction='DESCENDING').numpy()
    recommended_movies = [all_movie_titles[i] for i in top_indices if all_movie_titles[i] not in movie_list][:top_k]
    
    return recommended_movies


def terminal_controller():
    list_of_movies = []
    while True:
        print ("Welcome to the movie recommender system!")
        print ("Please enter what you want to do:")
        print ("1. Add a movie you like")
        print ("2. Get movie recommendations")
        print ("3. List your movies")
        print ("4. Clear screen")
        print ("5. Exit")
        print ("")
        sys.stdout.flush() 
        user_input = input("Enter your choice: ")
        if user_input == "1":
            movie = input("Enter the name of the movie you like: ")
            list_of_movies.append(movie)
            print (f"Added {movie} to your list of liked movies.")
            print ("")
        elif user_input == "2":
            recommendations = recommend_movies(model, list_of_movies, movies, top_k=5)
            print ("We recommend the following movies:")
            for i, movie in enumerate(recommendations):
                print (f"{i+1}. {movie}")
            print ("")
        elif user_input == "3":
            print ("You have liked the following movies:")
            for i, movie in enumerate(list_of_movies):
                print (f"{i+1}. {movie}")
            print ("")
        elif user_input == "4":
            clear_output()
        elif user_input == "5":
            print ("Thank you for using the movie recommender system!")
            break
        else:
            print ("Invalid input. Please try again.")
            print ("")


terminal_controller()
