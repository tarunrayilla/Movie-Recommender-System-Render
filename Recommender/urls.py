from django.urls import path
from . import views
from .models import Movie

urlpatterns = [
    path('', views.home, name='home'),
    path('<int:movieId>', views.rating, name='rating'),
    path('watched/', views.watched, name='watched')
]