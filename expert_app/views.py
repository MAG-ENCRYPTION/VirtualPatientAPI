from django.shortcuts import render, redirect
from learner_app.models import System
from patient_app.views import Convert1

from rest_framework import status, viewsets
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.parsers import FileUploadParser
from rest_framework.views import APIView

import json
import time
import os

from tutor_app.views import similar

from .models import *
from .serializers import *

import pandas as pd
import numpy as np
import pyAgrum as gum

# Create your views here.
path = os.path.realpath(__file__)
print("Le chemin du script est : " + path)
print("")
print("Importing the Bayesian network")
#bn=gum.loadBN("/app/expert_app/media/Bayesian_network.bif")
bn=gum.loadBN("expert_app\media\Bayesian_network.bif")
print("")
print("Successful import")
print("")
list_symptoms = [' congestion', ' belly_pain', ' phlegm', ' sinus_pressure', ' continuous_sneezing', 
' abdominal_pain', ' high_fever', ' receiving_blood_transfusion', ' yellowing_of_eyes', ' vomiting', ' palpitations', 
' blurred_and_distorted_vision', ' redness_of_eyes', ' muscle_pain', ' diarrhoea', ' red_spots_over_body', ' sweating', 
' irritability', ' toxic_look_typhos', ' mild_fever', ' swelled_lymph_nodes', ' constipation', ' slurred_speech', ' chest_pain', 
' breathlessness', ' cough', ' receiving_unsterile_injections', ' weight_loss', ' runny_nose', ' nausea', ' skin_rash', ' anxiety', 
' chills', ' excessive_hunger', ' lethargy', ' yellowish_skin', ' fast_heart_rate', ' loss_of_smell', ' loss_of_appetite', 'itching', 
' rusty_sputum', ' drying_and_tingling_lips', ' fatigue', ' yellow_urine', ' blood_in_sputum', ' joint_pain', ' dark_urine', ' headache', 
' throat_irritation', ' malaise']

def remove(symptoms):
  li_symp = []
  for symp in symptoms:
    li_symp.append(symp.replace(" ", ""))
  return li_symp

list_symptoms = remove(list_symptoms)
print("Remove space successful expert")

list_diseases = ['Malaria', 'Tuberculosis', 'Diabetes_', 'Typhoid', 'hepatitis_A', 'Hepatitis_B', 'Hypoglycemia', 'Common_Cold', 
'Chicken_pox', 'Pneumonia']

def infere_network(disease, symptoms):
  global bn
  ie=gum.LazyPropagation(bn)

  ie.setEvidence(symptoms)
  ie.makeInference()
  return ie.posterior(disease)

def getSymptom1(disease):
  global bn
  symps = bn.cpt(bn.idFromName(disease)).var_names.pop()

  return set(symps)

@api_view(['POST'])
def inference_disease_symptoms(request):
  start = time.time()
  global list_symptoms

  res = {}
  symp = {}
  body = json.loads(request.body)
  
  if body == None:
    result = {
      "status": "FAILURE",
      "message": "disease or symptoms required"
    }
    end= time.time()
    print("Temps" + str(end - start))
    return Response(result, status.HTTP_204_NO_CONTENT)
  
  else:
    new_symptoms = {}
    symptoms = body['symptoms']
    for key in symptoms:
      for symp in list_symptoms:
        if similar(key, symp) > 0.8:
          new_symptoms[symp] = symptoms[key]
          break

    
    if 'disease' not in body:
      for disease in list_diseases:
        res[disease] = infere_network(disease, new_symptoms)[1]

      dic = Convert1(sorted(res.items(), key=lambda x: x[1], reverse=True), {})
      res = dict(list(dic.items())[0: 5])
      end1 = time.time()
      print("Temps" + str(end1 - start))
      return Response(res, status=status.HTTP_200_OK)
    else:
      disease = body['disease']
      for dis in disease:
        res[dis] = infere_network(dis, new_symptoms)[1]

      end2 = time.time()  
      print("Temps" + str(end2 - start))    
      return Response(res, status=status.HTTP_200_OK)

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
    return Response(result, status=status.HTTP_404_NOT_FOUND)


def root(request):
    return redirect('/api')