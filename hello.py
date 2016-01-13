import graphlab as gl

# import data
products = gl.SFrame.read_csv('/Volumes/data/python/coursera/data/amazon_baby.csv')

# some data exploration
# print(products.head(n=10))

# add column to count number of words
products['word_count'] = gl.text_analytics.count_words(products['review'])

# explore the most popular product
top_product_reviews = products[products['name'] == "Vulli Sophie the Giraffe Teether"]
# top_product_reviews['rating'].show(view='Categorical')

# ignore all 3* reviews
products = products[products['rating'] != 3]

# positive sentiment = 4* or 5*
products['sentiment'] = products['rating'] >= 4

# training sentiment classifier
train_data, test_data = products.random_split(.8, seed=0)
sentiment_model = gl.logistic_classifier.create(train_data,target='sentiment',features=['word_count'],validation_set=test_data)

# evaluating classifier
sentiment_model.evaluate(test_data, metric='roc_curve')
sentiment_model.show('Evaluation')

# applying model
top_product_reviews['predicted_sentiment'] = sentiment_model.predict(top_product_reviews,output_type='probability')
top_product_reviews = top_product_reviews.sort('predicted_sentiment', ascending=0)
top_product_reviews.head()
top_product_reviews[0]['review']
top_product_reviews[-1]['review']