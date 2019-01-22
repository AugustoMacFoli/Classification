from django.urls import path
from . import views

urlpatterns = [
    path('', views.knn, name='knn'),
    path('knn_play', views.knn_play, name='knn_play'),
]