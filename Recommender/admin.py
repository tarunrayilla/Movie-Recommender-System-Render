from django.contrib import admin
from .models import Movie, Rating, newUser, Watched
# Register your models here.

admin.site.register(Movie)
admin.site.register(Rating)
admin.site.register(newUser)
admin.site.register(Watched)