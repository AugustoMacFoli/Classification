from django.urls import path
from . import views

urlpatterns = [
    path('', views.naive_bayes, name='naive_bayes'),
    path('naive_bayes_play', views.naive_bayes_play, name='naive_bayes_play'),
]