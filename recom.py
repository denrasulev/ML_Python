# building recommender system

import graphlab as gl

# read data
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
# every user gets the same recommendation, based on popularity of the product

# personalised model
personalised_model = gl.item_similarity_recommender.create(train_data,user_id='user_id',item_id='song')

# predictions
personalised_model.recommend(users=[users[0]])
personalised_model.recommend(users=[users[1]])

# similar items
personalised_model.get_similar_items(['With Or Without You - U2'])

# recommender models comparison

model_performance = gl.compare(test_data, [popularity_model, personalised_model], user_sample=0.05)
gl.show_comparison(model_performance,[popularity_model, personalised_model])

# assignment

# unique listeners to certain artists
len(song_data[song_data['artist'] == 'Kanye West']['user_id'].unique())
len(song_data[song_data['artist'] == 'Foo Fighters']['user_id'].unique())
len(song_data[song_data['artist'] == 'Taylor Swift']['user_id'].unique())
len(song_data[song_data['artist'] == 'Lady GaGa']['user_id'].unique())

# most and least popular artist
aggregated = song_data.groupby(key_columns='artist', operations={'total_count': gl.aggregate.SUM('listen_count')})
aggregated.sort('total_count', ascending=False)
aggregated.sort('total_count')

