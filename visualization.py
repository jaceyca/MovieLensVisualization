
import numpy as np
import matplotlib.pyplot as plt
import codecs
from collections import Counter

def main():
	# Y_train = np.loadtxt('./data/train.txt').astype(int)
 #    Y_test = np.loadtxt('./data/test.txt').astype(int)
    data = np.loadtxt('./data/data.txt').astype(int)
    print(data[:, 2])
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
    r1_movies, r2_movies, r3_movies, r4_movies, r5_movies = [],[],[],[],[]
    avg_ratings = {}
    for data_tuple in data:
    	key = data_tuple[1]
    	avg_ratings[key] = avg_ratings.get(key, 0) + data_tuple[2]/frequencies[key]
    	# Divide movies my rating
    	if data_tuple[2] == 1: r1_movies.append(data_tuple[1])
    	elif data_tuple[2] == 2: r2_movies.append(data_tuple[1])
    	elif data_tuple[2] == 3: r3_movies.append(data_tuple[1])
    	elif data_tuple[2] == 4: r4_movies.append(data_tuple[1])
    	elif data_tuple[2] == 5: r5_movies.append(data_tuple[1])

    # Plot histogram
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    width = 0.35  
    ind = np.arange(5) # x locations for the ratings
    rating_count = [len(r1_movies), len(r2_movies), 
                          len(r3_movies), len(r4_movies), len(r5_movies)] # frequencies for histogram
    rectangles = ax.bar(ind, rating_count, width, color='black')

    ax.set_xlim(-width,len(ind)-width)
    ax.set_ylim(0,max(rating_count))
    ax.set_xlabel('Rating')
    ax.set_ylabel('Number of Movies')
    ax.set_title('Ratings of All Movies')
    xTickMarks = [str(i) for i in range(1,6)]
    ax.set_xticks(ind)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, fontsize=10)
    fig.show()
    

    # 2. Ten most popular movies
    most_reviewed = frequencies.most_common(10)
    pop_movie_IDs = [x[0] for x in most_reviewed]
    pop_movie_names = [movie_names[ID] for ID in pop_movie_IDs]
    pop_movie_genres = [genres[ID] for ID in pop_movie_IDs]
    print("Most Reviewed Movies: ", pop_movie_names)

    # plot the histogram for the ratings of the most popular
    pop_r1, pop_r2, pop_r3, pop_r4, pop_r5 = 0, 0, 0, 0, 0
    for rating in data: 
        if rating[1] in pop_movie_IDs: 
            if rating[2] == 1: pop_r1 += 1
            elif rating[2] == 2: pop_r2 += 1
            elif rating[2] == 3: pop_r3 += 1
            elif rating[2] == 4: pop_r4 += 1
            elif rating[2] == 5: pop_r5 += 1

    pop_fig = plt.figure(2)
    ax_pop = pop_fig.add_subplot(111)
    pop_rating_count = [pop_r1, pop_r2, pop_r3, pop_r4, pop_r5]
    rectangles = ax_pop.bar(ind, pop_rating_count, width, color='black')

    ax_pop.set_xlim(-width,len(ind)-width)
    ax_pop.set_ylim(0,max(pop_rating_count))
    ax_pop.set_xlabel('Rating')
    ax_pop.set_ylabel('Number of Movies')
    ax_pop.set_title('Ratings of Ten Most Popular Movies')
    xTickMarks = [str(i) for i in range(1,6)]
    ax_pop.set_xticks(ind)
    xtickNames = ax_pop.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, fontsize=10)
    plt.show()


    # 3. Top ten best movies
    best_reviewed = dict(Counter(avg_ratings).most_common(10))
    best_reviewed_names = [movie_names[ID] for ID in best_reviewed]
    best_reviewed_genres = [genres[ID] for ID in best_reviewed]
    print("Best Movies: ", best_reviewed_names)

    # 4. Three genres of your choice - 2:Action, 7:Documentary, 17:War
    action_movies = [ID for ID in genres if genres[ID][2] == 1]
    action_movie_names = [movie_names[ID] for ID in action_movies] 

    documentary_movies = [ID for ID in genres if genres[ID][7] == 1]
    documentary_movie_names = [movie_names[ID] for ID in documentary_movies] 

    war_movies = [ID for ID in genres if genres[ID][17] == 1]
    war_movie_names = [movie_names[ID] for ID in war_movies] 

if __name__ == "__main__":
    main()