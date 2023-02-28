from django.db import models
from django.contrib.auth.models import User
from Recommender.models import newUser

# Create your models here.

class RegForm(models.Model):
    user = models.ForeignKey(newUser, on_delete=models.CASCADE)
    age = models.IntegerField()
    gender = models.CharField(max_length=10)
    genres = models.CharField(max_length=20)   
    movies = models.TextField(default="")
