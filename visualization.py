
import numpy as np
import matplotlib.pyplot as plt
import codecs
from collections import Counter

def main():
	# Y_train = np.loadtxt('./data/train.txt').astype(int)
 #    Y_test = np.loadtxt('./data/test.txt').astype(int)
    data = np.loadtxt('./data/data.txt').astype(int)
    movie_file = codecs.open('./data/movies.txt', mode='r', encoding='windows-1252')
    movie_names = {}
    genres = {}
    for line in movie_file:
    	movie_info = line.split()
    	movie_names[int(movie_info[0])] = " ".join(movie_info[1:-19])
    	genres[int(movie_info[0])] = list(map(int, movie_info[-19:]))

    M = max(data[:,0]).astype(int) # users
    N = max(data[:,1]).astype(int) # movies

    # 1. All movies
    frequencies = Counter(data[:,1]) # how often the movies are reviewed


    # 2. Ten most popular movies
    most_reviewed = frequencies.most_common(10)
    pop_movie_IDs = [x[0] for x in most_reviewed]
    pop_movie_names = [movie_names[ID] for ID in pop_movie_IDs]
    pop_movie_genres = [genres[ID] for ID in pop_movie_IDs]
    print("Most Reviewed Movies: ", pop_movie_names)

    # 3. Top ten best movies
    avg_ratings = {}
    for data_tuple in data:
    	key = data_tuple[1]
    	avg_ratings[key] = avg_ratings.get(key, 0) + data_tuple[2]/frequencies[key]
    best_reviewed = dict(Counter(avg_ratings).most_common(10))
    best_reviewed_names = [movie_names[ID] for ID in best_reviewed]
    best_reviewed_genres = [genres[ID] for ID in best_reviewed]
    print("Best Movies: ", best_reviewed_names)

    # 4. Three genres of your choice

if __name__ == "__main__":
    main()