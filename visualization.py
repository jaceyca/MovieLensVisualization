import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import codecs
import operator
from collections import Counter


# NOTE: Since show is a blocking function, it won't show the next graph until
# you've exited out of your current one
def bar_plot(rating_count, title):
    '''
    This function plots the rating count given in a bar graph

    Input: 
        rating_count: a list containing the number of each rating
        title: what the title of the plot should be
    Output: 
        shows the plot
        saves the plot under the title given 
    '''
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    width = 0.35  
    ind = np.arange(5) # x locations for the ratings

    rectangles = ax.bar(ind, rating_count, width, color='black')

    ax.set_xlim(-width,len(ind)-width)
    ax.set_ylim(0,max(rating_count))
    ax.set_xlabel('Rating')
    ax.set_ylabel('Number of Movies')
    ax.set_title(title)
    xTickMarks = [str(i) for i in range(1,6)]
    ax.set_xticks(ind)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, fontsize=10)
    plt.savefig(title)
    plt.show()

def categorize(genre_dict, movieIDs): 
    '''
    This function gives coordinates for each movie given the data we read in 
    and some iterable item containing all the ID's of the desired movies

    Input: 
        genre_dict: dictionary of the lists of genre specifcations
        movieIDs: the IDs of the movies we are considering

    Output: 
        a list of coordinates categorizing how serious(+)/escapist(-) and 
        aldrenaline-rush(+)/light-hearted(-) the movie is
    '''
    
    ###
    # 0-Unknown, 1-Action, 2-Adventure, 3-Animation, 4-Childrens, 5-Comedy, 
    # 6-Crime, 7-Documentary, 8-Drama, 9-Fantasy, 10-Film-Noir, 11-Horror, 
    # 12-Musical, 13-Mystery, 14-Romance, 15-Sci-Fi, 16-Thriller, 17-War, 18-Western
    ###
    coordinates = []
    for ID in movieIDs:
        fields = genre_dict[ID]

        # Serious = [Crime, Documentary, Drama, Film-noir, Mystery, War]
        # Escapist = [Fantasy, Horror, Romance, Sci-Fi, Thriller, Western]
        # Adrenaline-Rush = [Action, Adventure, Crime, Horror, Mystery, Thriller]
        # Light-Hearted = [Animation, Childrens, Comedy, Documentary, Musical, Romance]

        get_serious = operator.itemgetter(6,7,8,10,13,17)
        get_escapist = operator.itemgetter(9,11,14,15,16,18)
        get_adrenaline = operator.itemgetter(1,2,6,11,13,16)
        get_light = operator.itemgetter(3,4,5,7,12,14)

        serious_count = sum(get_serious(fields))
        escapist_count = sum(get_escapist(fields))
        adrenaline_count = sum(get_adrenaline(fields))
        light_count = sum(get_light(fields))

        coordinates.append((serious_count-escapist_count, adrenaline_count-light_count))

    return coordinates

def fancy_plot(genre_dict, movieIDs, movieNames, title):
    '''
    This function plots the rating count given in a bar graph

    Input: 
        genre_dict: dictionary of the lists of genre specifcations
        movieIDs: the IDs of the movies we are considering
        movieNames: the names of the movies we are considering
        title: what the title of the plot should be
    Output: 
        shows the plot
        saves the plot under the title given 
    '''

    # Fancy visualization
    coordinates = categorize(genre_dict, movieIDs)
    X = [x[0] for x in coordinates]
    Y = [x[1] for x in coordinates]
    
    fig = pl.figure()
    ax = fig.add_subplot(111)
    ax.plot(X, Y,'ko',markersize=8)
    lim = max([max(X), max(Y), abs(min(X)), abs(min(Y))])

    xmin, xmax = (-lim-0.5,lim+0.5)
    ymin, ymax = (-lim-0.5,lim+0.5)
    ax.set_xlim([-lim-0.5,lim+0.5])
    ax.set_ylim([-lim-0.5,lim+0.5])

    # plot movie names
    for label, xpt, ypt in zip(movieNames, X, Y):
        ax.text(xpt-0.1, ypt+0.1, label)

    # plot classifications
    ax.text(-0.4, ymax+0.5, 'Serious')
    ax.text(-0.4, ymin-0.5, 'Escapist')
    ax.text(xmax-0.6, 0.5, 'Adrenaline-Rush')
    ax.text(xmin-0.4, 0.5, 'Light-Hearted')

    # removing the default axis on all sides:
    for side in ['bottom','right','top','left']:
        ax.spines[side].set_visible(False)
 
    # removing the axis ticks
    pl.xticks([]) # labels 
    pl.yticks([])
    ax.xaxis.set_ticks_position('none') # tick markers
    ax.yaxis.set_ticks_position('none')

    # get width and height of axes object to compute 
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height
 
    # manual arrowhead width and length
    hw = 1./20.*(ymax-ymin) 
    hl = 1./20.*(xmax-xmin)
    lw = 1. # axis line width
    ohg = 0.3 # arrow overhang
 
    # compute matching arrowhead length and width
    yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width 
    yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height
 
    # draw x and y axis
    ax.arrow(0, 0, xmax, 0, fc='k', ec='k', lw = lw, 
         head_width=hw, head_length=hl, overhang = ohg, 
         length_includes_head= True, clip_on = False) 
    ax.arrow(0, 0, 0, ymax, fc='k', ec='k', lw = lw, 
         head_width=yhw, head_length=yhl, overhang = ohg, 
         length_includes_head= True, clip_on = False) 
    ax.arrow(0, 0, xmin, 0, fc='k', ec='k', lw = lw, 
         head_width=yhw, head_length=yhl, overhang = ohg, 
         length_includes_head= True, clip_on = False) 
    ax.arrow(0, 0, 0, ymin, fc='k', ec='k', lw = lw, 
         head_width=yhw, head_length=yhl, overhang = ohg, 
         length_includes_head= True, clip_on = False) 

    pl.savefig(title)
    pl.show()

def get_rating_count(data, movieIDs):
    '''
    This function gets the rating count given the data we read in and some
    iterable item containing all the ID's of the movies we want to tally

    Input: 
        data: the rating data that we read in 
        movieIDs: the IDs of the movies we are considering for the tally

    Output: 
        a list of how many ratings of each there were, from 1 to 5
    '''
    r1, r2, r3, r4, r5 = 0, 0, 0, 0, 0
    for rating in data: 
        if rating[1] in movieIDs: 
            if rating[2] == 1: r1 += 1
            elif rating[2] == 2: r2 += 1
            elif rating[2] == 3: r3 += 1
            elif rating[2] == 4: r4 += 1
            elif rating[2] == 5: r5 += 1

    return [r1, r2, r3, r4, r5]


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
    # fig = plt.figure(1)
    # ax = fig.add_subplot(111)
    # width = 0.35  
    # ind = np.arange(5) # x locations for the ratings
    # rating_count = [len(r1_movies), len(r2_movies), 
    #                       len(r3_movies), len(r4_movies), len(r5_movies)] # frequencies for histogram
    # rectangles = ax.bar(ind, rating_count, width, color='black')

    # ax.set_xlim(-width,len(ind)-width)
    # ax.set_ylim(0,max(rating_count))
    # ax.set_xlabel('Rating')
    # ax.set_ylabel('Number of Movies')
    # ax.set_title('Ratings of All Movies')
    # xTickMarks = [str(i) for i in range(1,6)]
    # ax.set_xticks(ind)
    # xtickNames = ax.set_xticklabels(xTickMarks)
    # plt.setp(xtickNames, fontsize=10)
    # fig.show()

    rating_count = [len(r1_movies), len(r2_movies), 
                          len(r3_movies), len(r4_movies), len(r5_movies)] # frequencies for histogram
    bar_plot(rating_count, 'Ratings of All Movies')
    

    # 2. Ten most popular movies
    most_reviewed = frequencies.most_common(10)
    pop_movie_IDs = [x[0] for x in most_reviewed]
    pop_movie_names = [movie_names[ID] for ID in pop_movie_IDs]
    pop_movie_genres = [genres[ID] for ID in pop_movie_IDs]
    print("Most Reviewed Movies: ", pop_movie_names)

    # Fancy plot
    fancy_plot(genres, pop_movie_IDs, pop_movie_names, 'Visualization of Ten Most Popular Movies')

    # Histogram
    pop_rating_count = get_rating_count(data, pop_movie_IDs)
    bar_plot(pop_rating_count, 'Ratings of Ten Most Popular Movies')


    # 3. Top ten best movies
    best_reviewed = dict(Counter(avg_ratings).most_common(10))
    best_reviewed_names = [movie_names[ID] for ID in best_reviewed]
    best_reviewed_genres = [genres[ID] for ID in best_reviewed]
    print("Best Movies: ", best_reviewed_names)


    best_rating_count = get_rating_count(data, best_reviewed)
    bar_plot(best_rating_count, 'Ratings of Ten Best Movies')

    # 4. Three genres of your choice - 2:Action, 7:Documentary, 17:War
    action_movies = [ID for ID in genres if genres[ID][2] == 1]
    action_movie_names = [movie_names[ID] for ID in action_movies] 

    # action movie plotting
    action_rating_count = get_rating_count(data, action_movies)
    bar_plot(action_rating_count, 'Ratings of Action Movies')

    documentary_movies = [ID for ID in genres if genres[ID][7] == 1]
    documentary_movie_names = [movie_names[ID] for ID in documentary_movies] 

    # plotting documentaries
    documentary_count = get_rating_count(data, documentary_movies)
    bar_plot(documentary_count, 'Ratings of Documentaries')

    war_movies = [ID for ID in genres if genres[ID][17] == 1]
    war_movie_names = [movie_names[ID] for ID in war_movies] 

    # plotting war movies
    war_rating_count = get_rating_count(data, war_movies)
    bar_plot(war_rating_count, 'Ratings of War Movies')


if __name__ == "__main__":
    main()