from django.urls import path, include, re_path
#from django.conf.urls import url
from rest_framework import routers

from . import views

routers = routers.DefaultRouter()
routers.register(r'doctors', views.DoctorViewSet)
routers.register(r'doctor-apprenant', views.LeanerPhysicianViewSet)
routers.register(r'doctor-expert', views.ExpertPhysicianViewSet)
routers.register(r'log', views.LogViewSet)
routers.register(r'virtuel-patient', views.VirtualPatientViewSet)
routers.register(r'clinical-case', views.ClinicalCaseViewSet)
routers.register(r'virtual-case', views.VirtualCaseViewSet)
routers.register(r'feedback', views.FeedbackViewSet)
routers.register(r'evaluation', views.EvaluationViewSet)
routers.register(r'diagnosis', views.DiagnosisViewSet)
routers.register(r'hypothesis', views.HypothesisViewSet)
routers.register(r'question', views.QuestionViewSet)
routers.register(r'personnal-information', views.PersonalInfoViewSet)
routers.register(r'ongoing-treatment', views.TreatmentInProgressViewSet)
routers.register(r'physical-diagnostic', views.PhysicalDiagnosisViewSet)
routers.register(r'exam', views.ExamViewSet)
routers.register(r'media', views.MediaViewSet)
routers.register(r'type-parametre', views.TypeParameterViewSet)
routers.register(r'medical-parameter', views.MedicalParameterViewSet)
routers.register(r'symptom', views.SymptomViewSet)
routers.register(r'life-style', views.LifeStyleViewSet)
routers.register(r'physical-activity', views.PhysicalActivityViewSet)
routers.register(r'addiction', views.AddictionViewSet)
routers.register(r'travel', views.TravelViewSet)
"""routers.register(r'concept', views.ConceptViewSet)"""
routers.register(r'medical-antecedent', views.MedicalAntecedentViewSet)
routers.register(r'obstetrical-antecedent', views.ObstetricalAntecedentViewSet)
routers.register(r'surgery', views.SurgeryViewSet)
routers.register(r'allergy', views.AllergyViewSet)
routers.register(r'treatment', views.TreatmentViewSet)
routers.register(r'disease', views.DiseaseViewSet)

urlpatterns = [

    path('', include(routers.urls)),
    path('getclinicalcase/<str:id_clinical_case>/',views.getClinicalCase, name='getClinicalCase'),
    path('getStat/',views.getStat, name='getStat'),
    path('save/',views.save, name='save'),
    path('getEvaluation/<str:id_evaluation>/',views.getEvaluation, name='getEvaluation'),
    path('getEvaluationByLearner/<str:id_learner>/',views.getEvaluationByLearner, name='getEvaluationByLearner'),
    path('getAllEvaluation/',views.getAllEvaluation, name='getAllEvaluation'),
    re_path(r'^[a-zA-Z0-9/,;:!\\*-+^$ù&é(-è_çà)]+/$', views.errorPage),
    
]