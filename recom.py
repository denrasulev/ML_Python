# building recommender system

import graphlab as gl

song_data = gl.SFrame.read_csv('/Volumes/data/python/coursera/data/song_data.csv')

print(song_data.head(n=10))

gl.canvas.set_target('browser')

song_data['song'].show()

# count number of users

users = song_data['user_id'].unique()

print(len(users))

# build recommender

train_data, test_data = song_data.random_split(0.8, seed=0)

# popularity based model

popularity_model = gl.popularity_recommender.create(train_data, user_id='user_id', item_id='song')

popularity_model.recommend(users=[users[0]])
popularity_model.recommend(users=[users[1]])
popularity_model.recommend(users=[users[1000]])
# every user gets the same recommendation, based on popularity of the product

# personalised model

personalised_model = gl.item_similarity_recommender.create(train_data,user_id='user_id',item_id='song')

personalised_model.recommend(users=[users[0]])
personalised_model.recommend(users=[users[1]])
personalised_model.recommend(users=[users[1000]])
personalised_model.get_similar_items(['With Or Without You - U2'])

