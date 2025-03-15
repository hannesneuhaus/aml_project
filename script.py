import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Dict, Text
import tensorflow_recommenders as tfrs
import sys
from IPython.display import clear_output

# This is the small dataset
ds_mov = 'Data/ml-latest-small/movies.csv'
ds_rat = 'Data/ml-latest-small/ratings.csv'
ds_tags = 'Data/ml-latest-small/tags.csv'

# This is the entire dataset, because of it's size this will take a long while to train the model
# ds_mov = 'Data/ml-latest/movies.csv'
# ds_rat = 'Data/ml-latest/ratings.csv'
# ds_tags = 'Data/ml-latest/tags.csv'


df_mov = pd.read_csv(ds_mov)
df_rat = pd.read_csv(ds_rat)
df_tags = pd.read_csv(ds_tags)
df_model = pd.merge(df_rat, df_mov, on='movieId', how='left')
df_model = pd.merge(df_model, df_tags, on=['movieId', 'userId'], how='left')
df_mov['movieId'] = df_mov['movieId'].astype(str)
df_model['userId'] = df_model['userId'].astype(str)
df_model['movieId'] = df_model['movieId'].astype(str)
df_model['tag'] = df_model['tag'].fillna('').astype(str)
df_model = df_model.drop(columns=['timestamp_x', 'timestamp_y'])


mov_feat = {key: tf.constant(value) for key, value in df_mov.to_dict(orient='list').items()}
movies = tf.data.Dataset.from_tensor_slices(mov_feat)

rat_feat = {key: tf.constant(value) for key, value in df_model.to_dict(orient='list').items()}
ratings = tf.data.Dataset.from_tensor_slices(rat_feat)

movies = movies.map(lambda x: x['title'])
model_data = ratings.map(lambda x: {'userId': x['userId'], 'title': x['title'], 'rating': x['rating'], 'tag': x['tag']})
shuffled = model_data.shuffle(100_000, seed=42, reshuffle_each_iteration=False)


class UserModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_user_ids.astype(str), mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding)
        ])
        self.tag_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_tags.astype(str), mask_token=None),
            tf.keras.layers.Embedding(len(unique_tags) + 1, embedding)
        ])
                
    def call(self, inputs):
        if isinstance(inputs, dict):
            user_embedding = self.user_embedding(inputs['userId'])
            tag_embedding = self.tag_embedding(inputs['tag'])
            return tf.concat([user_embedding, tag_embedding], axis=1)
        else:
            return self.user_embedding(inputs)


class MovieModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.movie_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_movie_titles.astype(str), mask_token=None),
            tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding*2)
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

        user_embeddings = self.user_model({'userId': features['userId'], 'tag': features['tag']})  
        movie_embeddings = self.movie_model(features['title'])

        return self.task(user_embeddings, movie_embeddings)

embedding = 32

unique_user_ids = np.unique([x.numpy().decode('utf-8') for x in model_data.map(lambda x: x['userId'])])
unique_movie_titles = np.unique([x.numpy().decode('utf-8') for x in movies])

unique_tags = np.unique([
    x.numpy().decode('utf-8') for x in model_data.map(lambda x: x['tag']) if x.numpy().decode('utf-8') != ''
])

model = RecommenderModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()

model.fit(cached_train, epochs=10)

def terminal_controller():
    list_of_movies = []
    while True:
        print ('Welcome to the movie recommender system!')
        print ('Please enter what you want to do:')
        print ('1. Add a movie you like')
        print ('2. Get movie recommendations')
        print ('3. List your movies')
        print ('4. Clear screen')
        print ('5. Exit')
        print ('')
        sys.stdout.flush() 
        user_input = input('Enter your choice: ')
        if user_input == '1':
            movie = input('Enter the name of the movie you like: ')
            list_of_movies.append(movie)
            print (f'Added {movie} to your list of liked movies.')
            print ('')
        elif user_input == '2':
            recommendations = recommend_movies(model, list_of_movies, movies, top_k=5)
            print ('We recommend the following movies:')
            for i, movie in enumerate(recommendations):
                print (f'{i+1}. {movie}')
            print ('')
        elif user_input == '3':
            print ('You have liked the following movies:')
            for i, movie in enumerate(list_of_movies):
                print (f'{i+1}. {movie}')
            print ('')
        elif user_input == '4':
            clear_output()
        elif user_input == '5':
            print ('Thank you for using the movie recommender system!')
            break
        else:
            print ('Invalid input. Please try again.')
            print ('')

terminal_controller()