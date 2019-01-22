from django.urls import path
from . import views

urlpatterns = [
    path('', views.logistic_regression, name='logistic_regression'),
    path('logistic_regression_play', views.logistic_regression_play, name='logistic_regression_play'),
]