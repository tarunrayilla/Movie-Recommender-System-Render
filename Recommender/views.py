from django.shortcuts import render
from joblib import load
from joblib import dump
import pickle
import math
import pandas as pd
import numpy as np
import sklearn
import requests
from django.contrib.auth.models import User
from sklearn import tree
from Recommender.models import MovieItem
from .models import Movie, Rating, newUser, Watched
from accounts.models import RegForm
from .forms import RatingForm 
from surprise import SVD
from surprise import Reader, Dataset
from django.templatetags.static import static
from surprise.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# Create your views here.

model = load('./mlModels/model.joblib')
model2 = load('./mlModels/model2.joblib')
matrix_id = pickle.load(open('./datafiles/matrix_id.pkl', 'rb'))
df_rating = pickle.load(open('./datafiles/df_rating.pkl', 'rb'))
links = pickle.load(open('./datafiles/links.pkl', 'rb'))
df_cold = pickle.load(open('./datafiles/df_cold.pkl', 'rb'))


def trainModel():
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df_rating, reader)
    trainData, testData = train_test_split(data, random_state=1)
    model = SVD(n_epochs=10, lr_all=0.005, reg_all=0.4)
    model.fit(trainData)
    dump(model, './mlModels/model.joblib')

def trainModel2(new_user_array, new_user):
    df_cold.loc[len(df_cold)] = new_user_array[0]
    df_cold = df_cold.rename(index={df_cold.index[len(df_cold)-1]: new_user.userId})
    data_cold = df_cold.values

    x = data_cold[:, :-1]
    y = data_cold[:, -1]

    model2 = tree.DecisionTreeClassifier()
    model2 = model2.fit(x, y)
    dump(model2, './mlModels/model2.joblib')

def recommend(uid):
    movies_unwatched = []
    temp = matrix_id.loc[uid]

    i = 0
    for a, b in temp.items(): 
        if math.isnan(b):
            movies_unwatched.append([a, round(model.predict(uid, a).est, 7)])
            #print([a, round(model.predict(uid, a).est, 7)])
            i+= 1

    print('i',i)        

    
    movies_unwatched = sorted(movies_unwatched, key = lambda x: x[1], reverse=True)

    recommendations = []
    for i in range(16):
        print(movies_unwatched[i][0])
        recommendations.append(links[links['movieId']==movies_unwatched[i][0]].tmdbId.item())
    
    print(recommendations)
    return recommendations

def getAllColdStartMovies(x_test, new_user):
    x_test = np.array([x_test])
    y_pred = model2.predict(x_test)
    new_user_array = np.append(x_test[0], y_pred)
    new_user_array = np.array([new_user_array])
    print(new_user_array)
    #trainModel2(new_user_array, new_user)

    rows = []
    for row in range(len(df_cold)):
        ind = df_cold.index[row]
        sim = cosine_similarity(np.array(df_cold.iloc[row, :]).reshape(1, 12), new_user_array)[0][0]
        rows.append([ind, sim])
        
    rows = sorted(rows, key = lambda x: x[1], reverse = True)
    #print(rows)

    n = 5
    similar_users = []
    temp = []
    for r in rows:
        if len(similar_users) == n:
            break
        
        if r not in similar_users:
            similar_users.append(r)
            temp.append(r[0])

    print(similar_users)

    similar_users = pd.DataFrame(similar_users, columns=['user_id', 'similarity'])
    similar_users = similar_users.set_index('user_id')
    print(similar_users)

    similar_user_movies = matrix_id[matrix_id.index.isin(temp)].dropna(axis=1, how='all')

    movie_score = {}
    for i in similar_user_movies.columns:
        movie_rating = similar_user_movies[i]
        total = 0
        count = 0
        for u in similar_users.index:
            if math.isnan(movie_rating[u]) is False:
                score = similar_users.loc[u].item() *  movie_rating[u]
                total += score
                count += 1
        
        movie_score[i] = total/count

    movie_score = pd.DataFrame(movie_score.items(), columns=['movie_id', 'score'])
    ranked_movie_score = movie_score.sort_values(by='score', ascending=False)
    m = 16
    # cold_movies = ranked_movie_score.head(m)

    temp = []

    for m in ranked_movie_score['movie_id']:
        if len(temp) ==  16:
            break
        tmdb_id = links[links['movieId']==m].tmdbId.item()
        if Watched.objects.filter(user=new_user, movie=tmdb_id).exists():
            print('already watched', tmdb_id)
            continue
        temp.append(m)

    recommendations = []
    for m in temp:
        if len(links[links['movieId']==m])== 0:
            print('hi')
            continue
        recommendations.append(links[links['movieId']==m].tmdbId.item())
    
        

    print(recommendations)
    return recommendations  

def recommendColdStart(new_user, movie_ids):
    recommendations = []

    for tmdb_id in movie_ids:
        if len(recommendations) ==  16:
            break
        if Watched.objects.filter(user=new_user, movie=tmdb_id).exists():
            print('already watched', tmdb_id)
            continue
        recommendations.append(tmdb_id)
    
    print(recommendations)
    return recommendations  

def fetchMovieDetails(movieId):
    print(movieId)
    movie = Movie.objects.filter(movieId = movieId)
    if not movie.exists():
        response = requests.get(f'https://api.themoviedb.org/3/movie/{movieId}?api_key=8e0c4c19d5e88ab122fde5feb28964bb&language=en-US')
        data = response.json()
        mid = data['id']
        title = data['title']
        if data['poster_path'] is not None:
            poster = "https://image.tmdb.org/t/p/w500/" + data['poster_path']
        else:
            poster = static('pics/default_poster.jpg') 
        genres_map = data['genres']
        genres = []
        for g in genres_map:
            genres.append(g['name'])
        genres = ', '.join(genres)    
        # movie = MovieItem(mid, title, poster, genres)
        movie = Movie(movieId = mid, title = title, poster = poster, genres = genres)
        movie.save()
        return movie
    else:
        movie = Movie.objects.get(movieId = movieId)
        return movie    


def home(request):
    if request.user.is_authenticated:
        # if newUser.objects.filter(user=request.user).exists():
        #     new_user = newUser.objects.get(user=request.user)
        # else:
        #     newId = newUser.objects.last().userId + 1
        #     new_user = newUser(userId=newId, user=request.user)    
        #     new_user.save()

        new_user = newUser.objects.get(user=request.user)

        uid = new_user.userId
        print('user Id', uid) 

        #add new user in user-item matrix
        if uid not in matrix_id.index:
            print('hello')
            matrix_id.loc[uid,:] = math.nan 
            pickle.dump(matrix_id, open('./datafiles/matrix_id.pkl', 'wb'))

        if Watched.objects.filter(user=new_user).count() > 10:
            # uid = new_user.userId
            # print('user Id', uid) 

            # #add new user in user-item matrix
            # if uid not in matrix_id.index:
            #     print('hello')
            #     matrix_id.loc[uid,:] = math.nan 
            #     pickle.dump(matrix_id, open('./datafiles/matrix_id.pkl', 'wb'))
            
        
            recommendations = recommend(uid)
            recommendedMovieDetails = []
            for mid in recommendations:
                movie = fetchMovieDetails(mid)
                #print('Movie Id', movie.movieId)
                recommendedMovieDetails.append(movie)
        
            return render(request, 'home.html', {'recommendations': recommendedMovieDetails})

        else:
            #cold start users
            reg_details = RegForm.objects.get(user=new_user)
            age = reg_details.age
            gender = reg_details.gender

            if gender == 'Male':
                gender = 1
            else:
                gender = 0    

            genres = reg_details.genres
            genres = genres.split('.')
            g = []
            for genre in genres:
                g.append(int(genre))

            x_test = g + [age] + [gender]
            print(x_test)

            #recommendations = getAllColdStartMovies(x_test, new_user)
            movie_ids = reg_details.movies
            movie_ids = movie_ids.split('.')
            recommendations = recommendColdStart(new_user, movie_ids)
            recommendedMovieDetails = []
            for mid in recommendations:
                movie = fetchMovieDetails(mid)
                #print('Movie Id', movie.movieId)
                recommendedMovieDetails.append(movie)

            return render(request, 'home.html', {'recommendations': recommendedMovieDetails})


    else:
        return render(request, 'home.html')    

def rating(request, movieId):
    movie = fetchMovieDetails(movieId)

    # form = RatingForm(request.POST or None)
    # #Rating
    # if request.method == "POST":
    #     rate = request.POST['rate']
    #     print(rate)
    #     print(type(rate))
    #     if newUser.objects.filter(user=request.user).exists():
    #         new_user = newUser.objects.get(user=request.user)
    #     else:
    #         new_user = newUser(user=request.user)    
    #         new_user.save()

    #     if Rating.objects.filter(user=new_user, movie=movieId).exists():
    #         Rating.objects.filter(user=new_user, movie=movieId).update(rating=rate)
    #     else:
    #         temp = Rating(user=new_user, movie=movie, rating=rate)
    #         temp.save()  



    # if newUser.objects.filter(user=request.user).exists():
    #     new_user = newUser.objects.get(user=request.user)
    # else:
    #     new_user = newUser(user=request.user)    
    #     new_user.save()

    new_user = newUser.objects.get(user=request.user)
    uid = new_user.userId

    if request.method == 'POST':

        if not Rating.objects.filter(user=new_user, movie=movieId).exists():
            temp = Rating(user=new_user, movie=movie)
            temp.save()

        if not Watched.objects.filter(user=new_user, movie=movieId).exists():
            temp = Watched(user=new_user, movie=movie)
            temp.save()    

        instance = Rating.objects.get(user=new_user, movie=movieId)    

        form = RatingForm(request.POST or None, instance=instance)
        print('Form data')
        print(form.data)
        if form.is_valid():
            form.save()

            instance = Rating.objects.get(user=new_user, movie=movieId)

            #find movie id in matrix corresponding to the tmdbid(movieId)
            mid = links[links['tmdbId']==movieId].movieId.item()

            #update rating in user-item matrix
            matrix_id.loc[uid][mid] = instance.rating
            pickle.dump(matrix_id, open('./datafiles/matrix_id.pkl', 'wb'))

            #Mark the movie as watched
            Watched.objects.filter(user=new_user, movie=movieId).update(watched = True)

            #Insert row in df_rating
            #row is present, so update the existing rating
            if len(df_rating.loc[(df_rating['user_id']==uid) & (df_rating['movie_id']==mid)]) == 1:
                df_rating.loc[(df_rating['user_id']==uid) & (df_rating['movie_id']==mid), 'rating'] = instance.rating
            else:
                df_rating.loc[len(df_rating.index)] = [uid, mid, instance.rating]

            pickle.dump(df_rating, open('./datafiles/df_rating.pkl', 'wb'))
            print('DF Rating',df_rating.tail(10))
            trainModel()    



        else:
            print('hi')  
    else:
        # form = RatingForm(instance=instance)
        if not Rating.objects.filter(user=new_user, movie=movieId).exists():
            instance = Rating(user=new_user, movie=movie, rating=0)
            print("GGGOOOOOKKUUU")
            print(instance)
            form = RatingForm(instance=instance, use_required_attribute=False)
            print(form.data)
            # form = RatingForm(instance=instance, use_required_attribute=False)
        else:
            instance = Rating.objects.get(user=new_user, movie=movieId)
            #form = RatingForm(instance=instance)

            form = RatingForm(instance=instance)

    return render(request, 'rating.html', {'movie': movie, 'form': form})

def watched(request):
    print('Wtacheddddd')
    new_user = newUser.objects.get(user=request.user)
    watched_objects = Watched.objects.filter(user=new_user)

    watched_movies = []
    for temp in watched_objects:
        watched_movies.append(temp.movie)

    return render(request, 'watched.html', {'watched_movies': watched_movies})
