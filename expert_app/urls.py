from django.urls import path, include, re_path
#from django.conf.urls import url
from rest_framework import routers

from . import views

routers = routers.DefaultRouter()

urlpatterns = [
    path('', include(routers.urls)),

    path('inference/',views.inference_disease_symptoms, name='inference_disease_symptoms'),
    re_path(r'^[a-zA-Z0-9/,;:!\\*-+^$ù&é(-è_çà)]+/$', views.errorPage),
    
]