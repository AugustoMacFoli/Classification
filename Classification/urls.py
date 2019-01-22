from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('', include('pages.urls')),
    path('logistic_regression/', include('logistic_regression.urls')),
    path('knn/', include('knn.urls')),
    path('svm/', include('svm.urls')),
    path('kernel_svm/', include('kernel_svm.urls')),
    path('naive_bayes/', include('naive_bayes.urls')),
	path('decision_tree/', include('decision_tree.urls')),
	path('random_forest/', include('random_forest.urls')),
    path('admin/', admin.site.urls),
]