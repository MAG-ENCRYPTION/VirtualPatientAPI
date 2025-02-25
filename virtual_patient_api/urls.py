"""virtual_patient_api URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include

from django.conf import settings
from django.conf.urls.static import static

from rest_framework.decorators import api_view
from rest_framework.response import Response
from patient_app import views as viewsPatient
from expert_app import views as viewsExpert
from learner_app import views as viewsLearner
from tutor_app import views as viewsTutor

from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)

@api_view(['GET'])
def root(request):
    """
        Entry point of the API
    """
    
    return Response({
        "USER LOGIN API": request.build_absolute_uri() + 'auth/login/',
        "USER REFRESH-TOKEN API": request.build_absolute_uri() + 'auth/refresh-token/',
        "VIRTUAL PATIENT API": request.build_absolute_uri() + 'virtual-patient/',
        "EXPERT MODULE": request.build_absolute_uri() + 'expert-module/',
        "LEARNER MODULE": request.build_absolute_uri() + 'learner-module/',
        "TUTOR MODULE": request.build_absolute_uri() + 'tutor-module/'
    })

urlpatterns = [
    path('api/', root),
    path('admin/', admin.site.urls),
    path('api/auth/login/', viewsPatient.CutomObtainPairView.as_view(), name='login'),
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('', viewsPatient.root),
    path('', viewsExpert.root),
    path('', viewsLearner.root),
    path('', viewsTutor.root),
    path('api/virtual-patient/', include('patient_app.urls')),
    path('api/expert-module/', include('expert_app.urls')),
    path('api/learner-module/', include('learner_app.urls')),
    path('api/tutor-module/', include('tutor_app.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
