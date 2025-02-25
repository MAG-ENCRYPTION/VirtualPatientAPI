from asyncio import constants
from audioop import avg
from pickle import FALSE
from unicodedata import decimal
from django.shortcuts import redirect
from patient_app.models import LeanerPhysician
from patient_app.serializers import LeanerPhysicianSerializer
from patient_app.views import Convert, similar
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.http.response import JsonResponse
from rest_framework.parsers import JSONParser
from django.db.models import Sum
import json
from decimal import Decimal


from .models import *
from .serializers import *

from math import sqrt

# Create your views here.

list_symptoms = [' congestion', ' belly_pain', ' phlegm', ' sinus_pressure', ' continuous_sneezing', 
' abdominal_pain', ' high_fever', ' receiving_blood_transfusion', ' yellowing_of_eyes', ' vomiting', ' palpitations', 
' blurred_and_distorted_vision', ' redness_of_eyes', ' muscle_pain', ' diarrhoea', ' red_spots_over_body', ' sweating', 
' irritability', ' toxic_look_typhos', ' mild_fever', ' swelled_lymph_nodes', ' constipation', ' slurred_speech', ' chest_pain', 
' breathlessness', ' cough', ' receiving_unsterile_injections', ' weight_loss', ' runny_nose', ' nausea', ' skin_rash', ' anxiety', 
' chills', ' excessive_hunger', ' lethargy', ' yellowish_skin', ' fast_heart_rate', ' loss_of_smell', ' loss_of_appetite', 'itching', 
' rusty_sputum', ' drying_and_tingling_lips', ' fatigue', ' yellow_urine', ' blood_in_sputum', ' joint_pain', ' dark_urine', ' headache', 
' throat_irritation', ' malaise']

class SymptomViewSet(viewsets.ModelViewSet):
  queryset = Symptom.objects.all()
  serializer_class = SymptomSerializer

class DiseaseViewSet(viewsets.ModelViewSet):
  queryset = Disease.objects.all()
  serializer_class = DiseaseSerializer

class WeightSymptomDiseaseViewSet(viewsets.ModelViewSet):
  queryset = WeightSymptomDisease.objects.all()
  serializer_class = WeightSymptomDiseaseSerializer

class SystemViewSet(viewsets.ModelViewSet):
  queryset = System.objects.all()
  serializer_class = SystemSerializer

class WeightDiseaseSystemViewSet(viewsets.ModelViewSet):
  queryset = WeightDiseaseSystem.objects.all()
  serializer_class = WeightDiseaseSystemSerializer

class RatingDiseaseViewSet(viewsets.ModelViewSet):
  queryset = RatingDisease.objects.all()
  serializer_class = RatingDiseaseSerializer

class RatingSystemViewSet(viewsets.ModelViewSet):
  queryset = RatingSystem.objects.all()
  serializer_class = RatingSystemSerializer

def max_similar1(keywords, word):
  max = 0
  maxWord = ''
  for key in keywords:
    sq = similar(key, word)
    if sq > max:
      max = sq
      maxWord = key
  
  return maxWord

def getRating(request, pk):
  systems = {}
  diseases = {}
  det = {}
  details = {}
  context = {
        'request': request,
    }

  try: 
    learner = LeanerPhysicianSerializer(LeanerPhysician.objects.get(pk=pk), many=False, context=context).data
  except LeanerPhysician.DoesNotExist: 
    return JsonResponse({'message': 'The learner does not exist'}, status=status.HTTP_404_NOT_FOUND)
  if request.method in ['GET', 'POST']:

    rating_system = RatingSystemSerializer(RatingSystem.objects.all().filter(learner=learner["id"]), many=True, context=context).data
    
    for sys in rating_system:
      sys = Convert(sys, {})
      s = sys["system"].split('/')
      syst = SystemSerializer(System.objects.get(pk=s[-2]), many=False, context=context).data
      systems[syst["name"]] = sys["rating"]

      det["rating"] = sys["rating"]
      diseases_rating = RatingDiseaseSerializer(RatingDisease.objects.all().filter(system=s[-2], learner=learner["id"]), many=True, context=context).data
      for dis_rate in diseases_rating:
        dis_rate = Convert(dis_rate, {})
        d = dis_rate["disease"].split('/')
        dis = DiseaseSerializer(Disease.objects.get(pk=d[-2]), many=False, context=context).data
        det[dis["name"]] = dis_rate["rating"]
      
      details[syst["name"]] = det

      det = {}

    
    #rating_disease = RatingDiseaseSerializer(RatingDisease.objects.all().filter(learner=learner["id"]), many=True, context=context).data
    rating_disease = list(Disease.objects.annotate(rating=Sum('ratingdisease__rating')).values('name', 'rating', 'ratingdisease__learner').filter(ratingdisease__learner=pk))
    
    for dis in rating_disease:
      diseases[dis["name"]] = dis["rating"]
    
  return systems, diseases, details

def getDate(request, pk):
  systems = {}
  det = {}
  details = {}
  context = {
        'request': request,
    }

  try: 
    learner = LeanerPhysicianSerializer(LeanerPhysician.objects.get(pk=pk), many=False, context=context).data
  except LeanerPhysician.DoesNotExist: 
    return JsonResponse({'message': 'The learner does not exist'}, status=status.HTTP_404_NOT_FOUND)
  if request.method == 'GET':

    rating_system = RatingSystemSerializer(RatingSystem.objects.all().filter(learner=learner["id"]), many=True, context=context).data

    for sys in rating_system:
      sys = Convert(sys, {})
      s = sys["system"].split('/')
      syst = SystemSerializer(System.objects.get(pk=s[-2]), many=False, context=context).data
      systems[syst["name"]] = sys["updated_at"]

      diseases_rating = RatingDiseaseSerializer(RatingDisease.objects.all().filter(system=s[-2], learner=learner["id"]), many=True, context=context).data
      for dis_rate in diseases_rating:
        dis_rate = Convert(dis_rate, {})
        d = dis_rate["disease"].split('/')
        dis = DiseaseSerializer(Disease.objects.get(pk=d[-2]), many=False, context=context).data
        det[dis["name"]] = dis_rate["updated_at"]
      
      details[syst["name"]] = det

      det = {}

    
    #rating_disease = RatingDiseaseSerializer(RatingDisease.objects.all().filter(learner=learner["id"]), many=True, context=context).data
    
    
  return systems, details

@api_view(['GET'])
def getratingUser(request, pk):
  """ systems = {}
  diseases = {}
  det = {}
  details = {}
  context = {
        'request': request,
    }

  try: 
    learner = LeanerPhysicianSerializer(LeanerPhysician.objects.get(pk=pk), many=False, context=context).data
  except LeanerPhysician.DoesNotExist: 
    return JsonResponse({'message': 'The learner does not exist'}, status=status.HTTP_404_NOT_FOUND)
  if request.method == 'GET':

    rating_system = RatingSystemSerializer(RatingSystem.objects.all().filter(learner=learner["id"]), many=True, context=context).data

    for sys in rating_system:
      sys = Convert(sys, {})
      s = sys["system"].split('/')
      syst = SystemSerializer(System.objects.get(pk=s[-2]), many=False, context=context).data
      systems[syst["name"]] = sys["rating"]

      det["rating"] = sys["rating"]
      diseases_rating = RatingDiseaseSerializer(RatingDisease.objects.all().filter(system=s[-2], learner=learner["id"]), many=True, context=context).data
      for dis_rate in diseases_rating:
        dis_rate = Convert(dis_rate, {})
        d = dis_rate["disease"].split('/')
        dis = DiseaseSerializer(Disease.objects.get(pk=d[-2]), many=False, context=context).data
        det[dis["name"]] = dis_rate["rating"]
      
      details[syst["name"]] = det

      det = {}

    
    #rating_disease = RatingDiseaseSerializer(RatingDisease.objects.all().filter(learner=learner["id"]), many=True, context=context).data
    rating_disease = list(Disease.objects.annotate(rating=Sum('ratingdisease__rating')).values('name', 'rating', 'ratingdisease__learner').filter(ratingdisease__learner=pk))
    
    for dis in rating_disease:
      diseases[dis["name"]] = dis["rating"] """
    
  systems, diseases, details = getRating(request=request, pk=pk)
  result = {
    "system":systems if systems else None,
    "disease":diseases if diseases else None,
    "detail":details if details else None
  }

  return Response(result, status=status.HTTP_200_OK)

@api_view(['POST'])
def rating(request):
  body = json.loads(request.body)
  global list_symptoms
  context = {
        'request': request,
    }

  if body == None:
    result = {
      "status": "FAILURE",
      "message": "learner, system, disease, symptoms, loss required"
    }
    return Response(result, status.HTTP_204_NO_CONTENT)
  else:
    if 'learner' not in body:
      result = {
        "status": "FAILURE",
        "message": "learner required"
      }
      return Response(result, status.HTTP_204_NO_CONTENT)
    if 'system' not in body:
      result = {
        "status": "FAILURE",
        "message": "system required"
      }
      return Response(result, status.HTTP_204_NO_CONTENT)
    if 'disease' not in body:
      result = {
        "status": "FAILURE",
        "message": "disease required"
      }
      return Response(result, status.HTTP_204_NO_CONTENT)
    if 'symptoms' not in body:
      result = {
        "status": "FAILURE",
        "message": "symptoms required"
      }
      return Response(result, status.HTTP_204_NO_CONTENT)
    if 'status' not in body:
      result = {
        "status": "FAILURE",
        "message": "status required"
      }
      return Response(result, status.HTTP_204_NO_CONTENT)
  
  try:
    learner = LeanerPhysician.objects.get(pk=body["learner"])
  except LeanerPhysician.DoesNotExist: 
    return JsonResponse({'message': 'The learner does not exist'}, status=status.HTTP_404_NOT_FOUND)
  
  disease = Disease.objects.get(name=body["disease"])
  print(body)
  if body["status"]:
    for system in body["system"] :
      sys = System.objects.get(name=system)
      #sys = System.objects.get(pk=system)
      rate_disease = 0
      new = FALSE
      try:
        weight_disease = Decimal(WeightDiseaseSystemSerializer(WeightDiseaseSystem.objects.get(disease=disease.id, system=sys.id), many=False, context=context).data["weight"])
      except WeightDiseaseSystem.DoesNotExist:
        continue
      
      for symptom in body["symptoms"]:
        nam = max_similar1(list_symptoms, symptom)
        print(nam)
        symptom = Symptom.objects.get(name=nam.replace(" ", ""))
        try:
          weight_symptom = WeightSymptomDiseaseSerializer(WeightSymptomDisease.objects.get(disease=disease.id, symptom=symptom.id), many=False, context=context).data
        except WeightSymptomDisease.DoesNotExist:
          continue
        
        rate_disease += Decimal(weight_symptom["weight"])
      try:
        rating_disease = RatingDisease.objects.get(disease=disease.id, system=sys.id, learner=body["learner"])
        rate = rating_disease.rating
        if rate == 0.0:
          rating_disease.rating = 0.1
        else:
          rating_disease.rating = round(sqrt(rate * rate_disease), 3)
        rating_disease.save()
      except RatingDisease.DoesNotExist:
        new = True
        rating_disease = RatingDisease.objects.create(
          rating = round(rate_disease, 3),
          disease = disease,
          learner = learner,
          system = sys
        )
        rating_disease.save()
        
      try:
        rating_system = RatingSystem.objects.get(system=sys.id, learner=body["learner"])
        if new:
          rating_system.rating += Decimal(rating_disease.rating) * weight_disease
        else:
          rating_system.rating -= Decimal(rate) * weight_disease
          rating_system.rating += round(Decimal(rating_disease.rating) * weight_disease, 3)
        rating_system.save()
      except RatingSystem.DoesNotExist:
        rating_system = RatingSystem.objects.create(
          rating = round(Decimal(rating_disease.rating) * weight_disease, 3),
          system = sys,
          learner = learner
        )
        rating_system.save()
  else:
    for system in body["system"] :
      sys = System.objects.get(name=system)
      new = FALSE
    
      try:
        rating_disease = RatingDisease.objects.get(disease=disease.id, system=sys.id, learner=body["learner"])
        rate = rating_disease.rating - Decimal(0.1)
        if rate <= 0.0:
          rating_disease.rating = Decimal(0.0)
        else:
          rating_disease.rating = round(Decimal(rate), 3)
        rating_disease.save()
      except RatingDisease.DoesNotExist:
        new = True
        
        rating_disease = RatingDisease.objects.create(
          rating = 0.000,
          disease = disease,
          learner = learner,
          system = sys
        )
        rating_disease.save()
      
      try:
        rating_system = RatingSystem.objects.get(system=sys.id, learner=body["learner"])
        if new:
          rating_system.rating += Decimal(rating_disease.rating) * weight_disease
        else:
          rating_system.rating -= Decimal(rate) * weight_disease
          rating_system.rating += round(Decimal(rating_disease.rating) * weight_disease, 3)
        rating_system.save()
      except RatingSystem.DoesNotExist:
        rating_system = RatingSystem.objects.create(
          rating = round(Decimal(rating_disease.rating) * weight_disease, 3),
          system = sys,
          learner = learner
        )
        rating_system.save()
  
  result = {
      "message": "rating success",
      "status": True
    }

  return Response(result, status=status.HTTP_200_OK)

@api_view(['GET'])
def errorPage(request):
    """
      This view is returned when no url matches the one called
    """
    result = {
      "status": False,
      "message": "Check your URL",
      "data": {}
    }
    return Response(result, status=status.HTTP_200_OK)


def root(request):
    return redirect('/api')