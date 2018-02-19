
import numpy as np
import matplotlib.pyplot as plt
import codecs
from collections import Counter

def main():
	# Y_train = np.loadtxt('./data/train.txt').astype(int)
 #    Y_test = np.loadtxt('./data/test.txt').astype(int)
    data = np.loadtxt('./data/data.txt').astype(int)
    # movies = np.genfromtxt('./data/movies.txt',dtype='str')
    movie_file = codecs.open('./data/movies.txt', mode='r', encoding='windows-1252')
    movie_names = {}
    ratings = {}
    for line in movie_file:
    	movie_info = line.split()
    	movie_names[int(movie_info[0])] = " ".join(movie_info[1:-19])
    	ratings[int(movie_info[0])] = list(map(int, movie_info[-19:]))

    M = max(data[:,0]).astype(int) # users
    N = max(data[:,1]).astype(int) # movies

    # All movies
    frequencies = Counter(data[:,1])

    # 2. Ten most popular movies
    most_reviewed = frequencies.most_common(10)
    pop_movie_IDs = [x[0] for x in most_reviewed]
    print(pop_movie_IDs)

    # 3. Top ten best movies
    # best_reviewed = 

if __name__ == "__main__":
    main()