from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MaxValueValidator, MinValueValidator

# Create your models here.

class MovieItem:
    def __init__(self, mid, title, poster, genres):
        self.mid = mid
        self.title = title
        self.poster = poster
        self.genres = genres

class Movie(models.Model):
    movieId = models.IntegerField(primary_key=True)          
    title = models.CharField(max_length=200)
    poster = models.URLField()
    genres = models.CharField(max_length=200)
 
class newUser(models.Model):
    # userId = models.IntegerField(primary_key=True)
    userId = models.IntegerField(primary_key=True)
    user =  models.ForeignKey(User, on_delete=models.CASCADE)

RATINGS = [
        (5, '5'),
        (4, '4'),
        (3, '3'),
        (2, '2'),
        (1, '1'),
    ]
    
# class Rating(models.Model):
#     user = models.ForeignKey(newUser, on_delete=models.CASCADE)
#     movie = models.ForeignKey(Movie, on_delete=models.CASCADE)
#     rating = models.IntegerField(default=0, validators=[MaxValueValidator(5), MinValueValidator(0)]) 

class Rating(models.Model):
    user = models.ForeignKey(newUser, on_delete=models.CASCADE)
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE)
    rating = models.IntegerField(default=0, choices=RATINGS) 

class Watched(models.Model):
    user = models.ForeignKey(newUser, on_delete=models.CASCADE)
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE)
    watched = models.BooleanField(default=False)
 