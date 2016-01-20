import graphlab as gl

people = gl.SFrame.read_csv('/Volumes/data/python/coursera/data/people_wiki.csv')

print(people.head(n=10))

len(people)

obama = people[people['name'] == 'Barack Obama']

obama['text']

# count words in Obama article

obama['word_count'] = gl.text_analytics.count_words(obama['text'])

# sort word counts

owct = obama[['word_count']].stack('word_count', new_column_name=['word', 'count'])

owct.sort('count', ascending=False)

# compute TF-IDF

people['word_count'] = gl.text_analytics.count_words(people['text'])

print(people.head())

tfidf = gl.text_analytics.tf_idf(people['word_count'])

people['tfidf'] = tfidf

print(people.head(n=10))

# examing TF-IDF

obama = people[people['name'] == 'Barack Obama']

obama[['tfidf']].stack('tfidf', new_column_name=['word','tfidf']).sort('tfidf', ascending=False)

# compute distances

clinton = people[people['name'] == 'Bill Clinton']
beckham = people[people['name'] == 'David Beckham']

gl.distances.cosine(obama['tfidf'][0], clinton['tfidf'][0])
gl.distances.cosine(obama['tfidf'][0], beckham['tfidf'][0])

# nearest neighbor model

knn_model = gl.nearest_neighbors.create(people,label='name',features=['tfidf'])

# apply model

knn_model.query(obama)

# other people

swift = people[people['name'] == 'Taylor Swift']

knn_model.query(swift)

# Assignment

elton = people[people['name'] == 'Elton John']

elton['word_count'] = gl.text_analytics.count_words(elton['text'])

elton_wc = elton[['word_count']].stack('word_count', new_column_name = ['word','count'])

print(elton_wc.sort('count',ascending=False).head())

# top tf-idf words for Elton John

elton[['tfidf']].stack('tfidf', new_column_name=['word','tfidf']).sort('tfidf', ascending=False)

# cosine distance between Elton and Victoria

victoria = people[people['name'] == 'Victoria Beckham']
victoria[['tfidf']].stack('tfidf', new_column_name=['word','tfidf']).sort('tfidf', ascending=False)

gl.distances.cosine(elton['tfidf'][0], victoria['tfidf'][0])

# cosine distance between Elton and Paul

paul = people[people['name'] == 'Paul McCartney']
gl.distances.cosine(elton['tfidf'][0],paul['tfidf'][0])