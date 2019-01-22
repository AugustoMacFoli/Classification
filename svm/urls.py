from django.urls import path
from . import views

urlpatterns = [
    path('', views.svm, name='svm'),
    path('svm_play', views.svm_play, name='svm_play'),
]