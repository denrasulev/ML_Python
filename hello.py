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
sentiment_model = gl.logistic_classifier.create(train_data,
                                                target='sentiment',
                                                features=['word_count'],
                                                validation_set=test_data)

# evaluating classifier
sentiment_model.evaluate(test_data, metric='roc_curve')
sentiment_model.show('Evaluation')

# applying model
top_product_reviews['predicted_sentiment'] = sentiment_model.predict(top_product_reviews, output_type='probability')
top_product_reviews = top_product_reviews.sort('predicted_sentiment', ascending=0)
top_product_reviews.head()


# function for counting key words
def word_counter(wrd, num):
    if wrd in num:
        return num[wrd]
    else:
        return 0

# set of selected words
selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']

# add them
for word in selected_words:
    products[word] = products['word_count'].apply(lambda word_dict: word_counter(word, word_dict))

# count these words
for word in selected_words:
    print "%s : %i" % (word, products[word].sum())

# split data to training and test sets
train_data,test_data = products.random_split(.8, seed=0)

# train model with selected words
selected_words_model = gl.logistic_classifier.create(train_data,
                                                     target='sentiment',
                                                     features=selected_words,
                                                     validation_set=test_data)

# sort by coefficient
selected_words_model = selected_words_model['coefficients'].sort('value')

# print them
selected_words_model.print_rows(num_rows=12)

# evaluate model
selected_words_model.evaluate(test_data, metric='roc_curve')

selected_words_model.show(view='Evaluation')

