from django.urls import path, include, re_path
#from django.conf.urls import url
from rest_framework import routers

from . import views

urlpatterns = [
    #path('', include(routers.urls)),

    path('response/',views.response, name='response'),
    path('hello_world/',views.hello_world, name='hello_world'),
    path('anamese/',views.anamnese_view, name='anamese'),
    path('selectClinicalCase/',views.selectClinicalCase, name='selectClinicalCase'),
    path('notate/',views.marking, name='marking'),
    path('verifysymptoms/',views.verifySymptom, name='verifySymptom'),
    re_path(r'^[a-zA-Z0-9/,;:!\\*-+^$ù&é(-è_çà)]+/$', views.errorPage),
    
]