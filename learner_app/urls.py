from django.urls import path, include, re_path
#from django.conf.urls import url
from rest_framework import routers

from . import views

routers = routers.DefaultRouter()
routers.register(r'symptom', views.SymptomViewSet)
routers.register(r'disease', views.DiseaseViewSet)
routers.register(r'weight-symptom-disease', views.WeightSymptomDiseaseViewSet)
routers.register(r'system', views.SystemViewSet)
routers.register(r'weight-disease-system', views.WeightDiseaseSystemViewSet)
routers.register(r'rating-disease', views.RatingDiseaseViewSet)
routers.register(r'rating-system', views.RatingSystemViewSet)

urlpatterns = [
    path('', include(routers.urls)),

    path('rating/<str:pk>/',views.getratingUser, name='getratingUser'),
    path('rating/',views.rating, name='rating'),
    re_path(r'^[a-zA-Z0-9/,;:!\\*-+^$ù&é(-è_çà)]+/$', views.errorPage),
    
]