from django.urls import path
from . import views

urlpatterns = [
    path('', views.kernel_svm, name='kernel_svm'),
    path('kernel_svm_play', views.kernel_svm_play, name='kernel_svm_play'),
]