from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.models import User, auth
from joblib import load
from joblib import dump
import pickle
import math
import numpy as np
import sklearn
import pandas as pd
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
import requests
from django.contrib.auth.models import User
from Recommender.models import newUser, Watched
from .models import RegForm
from sklearn.metrics.pairwise import cosine_similarity

model2 = load('./mlModels/model2.joblib')
matrix_id = pickle.load(open('./datafiles/matrix_id.pkl', 'rb'))
links = pickle.load(open('./datafiles/links.pkl', 'rb'))
df_cold = pickle.load(open('./datafiles/df_cold.pkl', 'rb'))

print(df_cold)


def login(request):
    if request.method== 'POST':
        username = request.POST['username']
        password = request.POST['password']

        user = auth.authenticate(username=username,password=password)

        if user is not None:
            auth.login(request, user)
            
            if newUser.objects.filter(user=request.user).exists():
                return redirect("/")
            else:
                newId = newUser.objects.last().userId + 1
                new_user = newUser(userId=newId, user=request.user)    
                new_user.save()
                return redirect('registration')

        else:
            messages.info(request,'Invalid Credentials')
            return redirect('login')

    else:
        return render(request,'login2.html')    

def signup(request):

    if request.method == 'POST':
        first_name = request.POST['first_name']
        last_name = request.POST['last_name']
        username = request.POST['username']
        password1 = request.POST['password1']
        password2 = request.POST['password2']
        email = request.POST['email']

        if password1==password2:
            if User.objects.filter(username=username).exists():
                messages.info(request,'Username Taken')
                return redirect('signup')
            elif User.objects.filter(email=email).exists():
                messages.info(request,'Email Taken')
                return redirect('signup')
            else:   
                user = User.objects.create_user(username=username, password=password1, email=email,first_name=first_name,last_name=last_name)
                user.save()
                print('user created')
                # return redirect('registration')
                return redirect('login')

        else:
            messages.info(request,'password not matching..')    
            return redirect('signup')
        return redirect('/')
        
    else:
        return render(request,'signup2.html')

def logout(request):
    auth.logout(request)
    return redirect('/')

def trainModel2(new_user_array, new_user):
    global df_cold
    print('TM2')
    print(df_cold)
    df_cold.loc[len(df_cold)] = new_user_array[0]
    df_cold = df_cold.rename(index={df_cold.index[len(df_cold)-1]: new_user.userId})
    data_cold = df_cold.values

    print(df_cold)

    x = data_cold[:, :-1]
    y = data_cold[:, -1]

    print(x)
    print(y)

    model2 = GradientBoostingClassifier()
    model2 = model2.fit(x, y)
    dump(model2, './mlModels/model2.joblib')
    pickle.dump(df_cold, open('./datafiles/df_cold.pkl', 'wb'))

    print('Trained Model 2')


def getAllColdStartMovies(x_test, new_user):
    x_test = np.array([x_test])
    y_pred = model2.predict(x_test)
    new_user_array = np.append(x_test[0], y_pred)
    new_user_array = np.array([new_user_array])
    print(new_user_array)

    #print(df_cold)

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
        
        # if r not in similar_users:
        #     similar_users.append(r)
        #     temp.append(r[0])

        if r[0] not in temp:
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
                print('SI', similar_users.loc[u])
                score = similar_users.loc[u].item() *  movie_rating[u]
                total += score
                count += 1
        
        movie_score[i] = total/count

    movie_score = pd.DataFrame(movie_score.items(), columns=['movie_id', 'score'])
    ranked_movie_score = movie_score.sort_values(by='score', ascending=False)
    m = 50
    # cold_movies = ranked_movie_score.head(m)

    temp = []

    for m in ranked_movie_score['movie_id']:
        if len(temp) ==  50:
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
        recommendations.append(str(int(links[links['movieId']==m].tmdbId.item())))
    

    trainModel2(new_user_array, new_user)    

    print(recommendations)
    return recommendations     

def registration(request):
    if request.method == 'POST':
        li = []
        age = request.POST['age']
        age = int(age)
        gender = request.POST['gender']
        if gender == 'Male':
            gender2 = 1
        else:
            gender2 = 0    
        li = request.POST.getlist('genres')
        print(age)
        print(gender)
        print(li)
        genres = ['0']*9
        for g in li:
            genres[int(g)] = '1'

        genres2 = []
        for genre in genres:
            genres2.append(int(genre))    

        genres = '.'.join(genres)    
        print(genres)

        print('Genres', genres2)
        print('age', age)
        print('gendr', gender2)

        x_test = genres2 + [age] + [gender2]
        print('x_test', x_test)
        new_user = newUser.objects.get(user=request.user)
        recommendations = getAllColdStartMovies(x_test, new_user)
        recommendations = '.'.join(recommendations)

        if not RegForm.objects.filter(user=new_user).exists():
            temp = RegForm(user=new_user, age=age, gender=gender, genres=genres, movies=recommendations)  
            temp.save()
            #return redirect('registration')

        return redirect('/')
 
    else:    
        return render(request, 'registration2.html')    