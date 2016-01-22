# building recommender

import graphlab as gl

song_data = gl.SFrame.read_csv('/Volumes/data/python/coursera/data/song_data.csv')

print(song_data.head(n=10))

gl.canvas.set_target('browser')

song_data['song'].show()

