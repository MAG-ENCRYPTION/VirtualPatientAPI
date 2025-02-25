from unittest import result
from django.shortcuts import render
import logging
import re
import pandas as pd
from learner_app.views import getDate, getRating
import numpy as np
from patient_app.models import ClinicalCase as CC, LeanerPhysician
from patient_app.serializers import ClinicalCaseSerializer, LeanerPhysicianSerializer
from patient_app.views import Convert1, clinicalCase, getClinicalCase, similar
#from expert_app.views import getSymptom1
from rest_framework import status, viewsets
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.parsers import FileUploadParser
from rest_framework.views import APIView
from django.shortcuts import redirect
from django.http.response import JsonResponse
from django.http import HttpRequest
# from transformers import GPT2LMHeadModel, GPT2Tokenizer

import requests as rq
import json
import ast
import time

from datetime import datetime, timezone
from decimal import Decimal
import pytz

from math import exp

from nltk import PCFG
from random import randint

import pyAgrum as gum

# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# import torch
from rest_framework.decorators import api_view
from rest_framework.response import Response
import logging


class VirtualPatient:
    def __init__(self, clinical_case):
        self.name = clinical_case["name"]
        self.age = clinical_case["age"]
        self.gender = clinical_case["gender"]
        self.symptoms = clinical_case["symptoms"]
        self.history = clinical_case["history"]
        self.diagnosis = clinical_case["diagnosis"]
        self.emotions = clinical_case["emotions"]
        self.pain_level = clinical_case["pain_level"]
        self.conversation_history = []

    def generate_response_prompt(self, question):
        prompt = f"The patient {self.name} is a {self.age}-year-old {self.gender} who is feeling {self.emotions}. "
        if "symptoms" in question.lower():
            prompt += f"They report experiencing {', '.join(self.symptoms)}."
        elif "history" in question.lower():
            prompt += f"They have a history of {self.history}."
        elif "pain" in question.lower():
            prompt += f"Their pain level is {self.pain_level}."
        else:
            prompt += f"They seem to be struggling with answering the question."
        
        return prompt


clinical_cases = {
    "case_1": {
        "name": "John Doe",
        "age": 45,
        "gender": "male",
        "symptoms": ["fever", "abdominal pain", "nausea"],
        "history": "type 2 diabetes",
        "diagnosis": "appendicitis",
        "emotions": "anxious and in pain",
        "pain_level": "high",
    },
    "case_2": {
        "name": "Jane Smith",
        "age": 30,
        "gender": "female",
        "symptoms": ["cough", "sore throat", "fatigue"],
        "history": "none",
        "diagnosis": "common cold",
        "emotions": "tired",
        "pain_level": "low",
    },
    "case_3": {
        "name": "Michael Johnson",
        "age": 60,
        "gender": "male",
        "symptoms": ["chest pain", "shortness of breath"],
        "history": "hypertension",
        "diagnosis": "myocardial infarction",
        "emotions": "fearful",
        "pain_level": "very high",
    },
    "case_4": {
        "name": "Emily Davis",
        "age": 25,
        "gender": "female",
        "symptoms": ["headache", "blurred vision"],
        "history": "migraine",
        "diagnosis": "migraine",
        "emotions": "irritated",
        "pain_level": "medium",
    },
    "case_5": {
        "name": "Robert Brown",
        "age": 50,
        "gender": "male",
        "symptoms": ["joint pain", "stiffness"],
        "history": "arthritis",
        "diagnosis": "osteoarthritis",
        "emotions": "frustrated",
        "pain_level": "medium",
    },
    
    "case_6": {
        "name": "Laura Wilson",
        "age": 40,
        "gender": "female",
        "symptoms": ["rash", "itchiness"],
        "history": "eczema",
        "diagnosis": "allergic reaction",
        "emotions": "uncomfortable",
        "pain_level": "low",
    },
    "case_7": {
        "name": "James Taylor",
        "age": 35,
        "gender": "male",
        "symptoms": ["back pain", "numbness"],
        "history": "disc herniation",
        "diagnosis": "sciatica",
        "emotions": "distressed",
        "pain_level": "high",
    },
    "case_8": {
        "name": "Olivia Martinez",
        "age": 28,
        "gender": "female",
        "symptoms": ["pelvic pain", "heavy bleeding"],
        "history": "none",
        "diagnosis": "endometriosis",
        "emotions": "worried",
        "pain_level": "high",
    },
    "case_9": {
        "name": "William Anderson",
        "age": 55,
        "gender": "male",
        "symptoms": ["fatigue", "weight gain"],
        "history": "none",
        "diagnosis": "hypothyroidism",
        "emotions": "lethargic",
        "pain_level": "low",
    },
    "case_10": {
        "name": "Sophia Thomas",
        "age": 22,
        "gender": "female",
        "symptoms": ["sneezing", "runny nose"],
        "history": "allergies",
        "diagnosis": "allergic rhinitis",
        "emotions": "bothered",
        "pain_level": "low",
    },
    
}


# Initialisation du logger
logger = logging.getLogger(__name__)

# Charger le modèle GPT-2 et le tokenizer une fois au démarrage du serveur
# model = GPT2LMHeadModel.from_pretrained('gpt2')
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# # Vérification de l'accès à un GPU pour des performances améliorées
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model.to(device)

# # Mettre le modèle en mode évaluation (désactive les dropouts)
# model.eval()

logger = logging.getLogger(__name__)

# Fonction simple pour répondre aux questions
def simple_response_generator(question):
    """
    Fonction pour générer une réponse simple basée sur des règles prédéfinies.
    """
    # Exemple de correspondance basée sur des mots-clés
    if re.search(r'\b(fatigue|tired|exhausted)\b', question, re.IGNORECASE):
        return "La fatigue peut être due à de nombreuses causes. Pouvez-vous fournir plus de détails ?"
    elif re.search(r'\b(fever|temperature|feverish)\b', question, re.IGNORECASE):
        return "La fièvre peut indiquer une infection. Depuis combien de temps cela dure-t-il ?"
    elif re.search(r'\b(pain|ache|hurt)\b', question, re.IGNORECASE):
        return "Pouvez-vous préciser l'emplacement et l'intensité de la douleur ?"
    else:
        return "Je ne suis pas sûr de comprendre votre question. Pouvez-vous clarifier ?"

@api_view(['GET', 'POST'])
def anamnese_view(request):
    if request.method == 'POST':
        user_input = request.data.get('question', '').strip()
        clinical_case_id = request.data.get('clinicalCaseId', '').strip()

        # Validation de l'entrée utilisateur
        if not user_input or not clinical_case_id:
            return Response({'error': 'Question et ID de cas clinique ne peuvent pas être vides'}, status=400)

        try:
            # Générer une réponse basée sur des règles simples
            generated_text = simple_response_generator(user_input)

            # Déterminer la classe de la séquence et le statut des symptômes
            response_class = "General"  # Remplace par une logique réelle pour déterminer la classe
            symptom_status = "Not Detected"  # Remplace par une logique réelle pour vérifier les symptômes

            # Structure de la réponse
            response_data = {
                "response": generated_text,
                "class": response_class,
                "status": symptom_status,
                "symptom": "N/A"  # Remplace par une logique réelle pour récupérer les symptômes
            }

            return Response(response_data)

        except Exception as e:
            # Journaliser l'erreur pour un débogage plus facile
            logger.error(f"Error processing the request: {e}")
            return Response({'error': str(e)}, status=500)

    return Response({"response_data": "Use POST method to interact with this endpoint."})


# @api_view(['GET', 'POST'])
# def anamnese_view(request):
#     if request.method == 'POST':
#         user_input = request.data.get('question', '').strip()
#         clinical_case_id = request.data.get('clinicalCaseId', '').strip()

#         # Validation de l'entrée utilisateur
#         if not user_input or not clinical_case_id:
#             return Response({'error': 'Question et ID de cas clinique ne peuvent pas être vides'}, status=400)

#         try:
#             # Tokenisation du texte de l'utilisateur
#             input_ids = tokenizer.encode(user_input, return_tensors='pt')
#             input_ids = input_ids.to(device)

#             # Génération de texte par le modèle GPT-2
#             with torch.no_grad():
#                 output = model.generate(
#                     input_ids,
#                     max_length=100,
#                     num_return_sequences=1,
#                     no_repeat_ngram_size=2,
#                     pad_token_id=tokenizer.eos_token_id
#                 )

#             # Décodage de la réponse générée
#             generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

#             # Déterminer la classe de la séquence et le statut des symptômes
#             response_class = "General"  # Remplace par une logique réelle pour déterminer la classe
#             symptom_status = False  # Remplace par une logique réelle pour vérifier les symptômes

#             # Structure de la réponse
#             response_data = {
#                 "response": generated_text,
#                 "class": response_class,
#                 "status": symptom_status,
#                 "symptom": "Symptom example"  # Remplace par une logique réelle pour récupérer les symptômes
#             }

#             return Response(response_data)

#         except Exception as e:
#             # Journaliser l'erreur pour un débogage plus facile
#             logger.error(f"Error processing the request: {e}")
#             return Response({'error': str(e)}, status=500)

#     return Response({"response_data": "Use POST method to interact with this endpoint."})

import random
# from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from rest_framework.decorators import api_view
from rest_framework.response import Response

# Define the lists of possible values
ages = [25, 35, 45, 55, 65, 75]
sexes = ['male', 'female']
symptoms_list = [
    ["severe fatigue", "intermittent chest pain", "shortness of breath"],
    ["persistent cough", "fever", "night sweats"],
    ["abdominal pain", "nausea", "vomiting"],
    ["headache", "dizziness", "blurred vision"]
]
family_histories = [
    "cardiovascular disease",
    "diabetes",
    "cancer",
    "no significant family history"
]
durations = ["two weeks", "one month", "several days"]
vital_findings = ["elevated temperature", "slightly elevated blood pressure", "normal vital signs"]
physical_exam_findings = [
    "mild edema in the lower extremities",
    "tenderness in the abdomen",
    "no significant abnormalities"
]
lab_findings = [
    "elevated cholesterol levels",
    "low hemoglobin levels",
    "elevated white blood cell count"
]
possible_diagnoses = [
    ["angina", "heart failure"],
    ["bronchitis", "pneumonia"],
    ["gastroenteritis", "peptic ulcer"],
    ["migraine", "vertigo"]
]

# Load the RoBERTa model and tokenizer
# model_name = "deepset/roberta-base-squad2"
# model = AutoModelForQuestionAnswering.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Create a question-answering pipeline
# qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)

def get_context():
    age = random.choice(ages)
    sex = random.choice(sexes)
    symptoms = random.choice(symptoms_list)
    duration = random.choice(durations)
    family_history = random.choice(family_histories)
    vital_finding = random.choice(vital_findings)
    physical_exam_finding = random.choice(physical_exam_findings)
    lab_finding = random.choice(lab_findings)
    diagnoses = random.choice(possible_diagnoses)

    context = f"""
    Hello, I am a {age}-year-old {sex} patient. 
    I have been experiencing {symptoms[0]}, {symptoms[1]}, and {symptoms[2]} for the past {duration}. 
    These symptoms have been progressively worsening and are affecting my daily activities. 
    My family has a history of {family_history}. 
    I have had no recent travel or contact with sick individuals. My vital signs are mostly normal except for {vital_finding}. 
    During the physical examination, {physical_exam_finding} was observed, and the initial lab results show {lab_finding}.

    Based on this, the doctor is considering diagnoses like {diagnoses[0]} or {diagnoses[1]}. 
    If you have any questions, I will be glad to provide more details. 
    If you ask about my medical history, previous treatments, allergies, chronic conditions, or specific pain locations, I will provide information accordingly. 
    If asked to wait for tests, I will say thank you.
    """

    return {
        'context': context,
        'symptoms': symptoms,
        'duration': duration,
        'family_history': family_history,
        'physical_exam_finding': physical_exam_finding,
        'lab_finding': lab_finding
    }

def ask_question(question, context_info):
    context = context_info['context']
    QA_input = {
        'question': question,
        'context': context
    }
    result = qa_pipeline(QA_input)
    
    # Check if the model is confident in its answer, otherwise return "I don't know"
    if result['score'] < 0.2:  # You can adjust this threshold
        return "I don't know"
    return result['answer']

@api_view(['POST'])
def patient_interaction(request):
    if request.method == 'POST':
        user_input = request.data.get('question', '').strip()

        # Validation de l'entrée utilisateur
        if not user_input:
            return Response({'error': 'Question cannot be empty'}, status=400)

        # Generate context and details
        context_info = get_context()

        # Handle specific questions
        if "hello" in user_input.lower():
            return Response({"response": f"Hi Doctor ! I'm not fine today ! Please Help me"})
        elif "what brings you here" in user_input.lower():
            return Response({
                "response": f"I have been experiencing {context_info['symptoms'][0]}, {context_info['symptoms'][1]}, and {context_info['symptoms'][2]} for the past {context_info['duration']}."
            })
        elif "medical history" in user_input.lower():
            return Response({"response": f"My family has a history of {context_info['family_history']}."})
        elif "previous treatments" in user_input.lower():
            return Response({"response": "I have not had any significant previous treatments."})
        elif "allergies" in user_input.lower():
            return Response({"response": "I have no known allergies."})
        elif "chronic conditions" in user_input.lower():
            return Response({"response": "I have no chronic conditions."})
        elif "pain location" in user_input.lower():
            return Response({"response": "I am experiencing pain in my abdomen."})
        elif "wait for tests" in user_input.lower():
            return Response({"response": "Thank you Doc ! "})
        else:
            # General question handling
            answer = ask_question(user_input, context_info)
            return Response({"response": answer})

    return Response({"response_data": "You are using the GET method"})


@api_view(['GET', 'POST'])
def hello_world(request):
    if request.method == 'POST':
        return Response({"message": "Got some data!", "data": request.data})
    return Response({"message": "Hello, world!"})



url_symptom = "https://50878.gradio.app/api/predict/"
session_symptom = "2dzf619vjvp"

url_life_style = "https://45044.gradio.app/api/predict/"
session_life_style = "po8r3opqd5"

url_antecedent = "https://14562.gradio.app/api/predict/"
session_antecedent = "cekrdgoffms"

url_classify = "https://49830.gradio.app/api/predict/"
session_classify = "x9f8opd6prm"

list_symptoms = [' congestion', ' belly_pain', ' phlegm', ' sinus_pressure', ' continuous_sneezing', 
' abdominal_pain', ' high_fever', ' receiving_blood_transfusion', ' yellowing_of_eyes', ' vomiting', ' palpitations', 
' blurred_and_distorted_vision', ' redness_of_eyes', ' muscle_pain', ' diarrhoea', ' red_spots_over_body', ' sweating', 
' irritability', ' toxic_look_typhos', ' mild_fever', ' swelled_lymph_nodes', ' constipation', ' slurred_speech', ' chest_pain', 
' breathlessness', ' cough', ' receiving_unsterile_injections', ' weight_loss', ' runny_nose', ' nausea', ' skin_rash', ' anxiety', 
' chills', ' excessive_hunger', ' lethargy', ' yellowish_skin', ' fast_heart_rate', ' loss_of_smell', ' loss_of_appetite', 'itching', 
' rusty_sputum', ' drying_and_tingling_lips', ' fatigue', ' yellow_urine', ' blood_in_sputum', ' joint_pain', ' dark_urine', ' headache', 
' throat_irritation', ' malaise']

#bn=gum.loadBN("/app/expert_app/media/Bayesian_network.bif")
bn=gum.loadBN("expert_app\media\Bayesian_network.bif")

def getSymptom1(disease):
  global bn
  symps = bn.cpt(bn.idFromName(disease)).names
  symps = set(symps)
  l = []

  for s in symps:
    if "_" in s :
      l.append(s.replace("_", " "))
    elif s!=disease:
      l.append(s)

  return set(l)

def remove(symptoms):
  li_symp = []
  for symp in symptoms:
    t = symp.replace(" ", "")
    if "_" in t:
      li_symp.append(t.replace("_", ""))
    else:
      li_symp.append(t)
  return li_symp

list_symptoms = remove(list_symptoms)
print("Remove space successful tutor")
# Create your views here.
def getkeySymptom(text):
  r = rq.post(url_symptom, json={"fn_index": 0, "data": [text], "session_hash": session_symptom})
  t = r.json()
  result = {
    "data":ast.literal_eval(t["data"][0])
  }
  return result

def getkeyLifeStyle(text):
  r = rq.post(url_life_style, json={"fn_index": 0, "data": [text], "session_hash": session_life_style})
  t = r.json()
  result = {
    "data":ast.literal_eval(t["data"][0])
  }
  return result

def getkeyAntecedent(text):
  r = rq.post(url_antecedent, json={"fn_index": 0, "data": [text], "session_hash": session_antecedent})
  t = r.json()
  result = {
    "data":ast.literal_eval(t["data"][0])
  }
  return result

def classifyText(text):
  r = rq.post(url_classify, json={"fn_index": 0, "data": [text], "session_hash": session_classify})
  t = r.json()

  return t["data"][0]

def getPeriod():
  now = datetime.now()
  hr = now.hour + 1 # GMT+1  in Cameroon
  if (hr >= 5 and hr < 12):
    return 'morning'
  elif (hr >= 12 and hr < 18):
    return 'afternoon'
  elif (hr >= 18 and hr < 22):
    return 'evening'
  else:
    return 'night'

def getSymptoms(clinical_case, symptom):
  symptoms = clinical_case['symptom']
  for symp in symptoms:
    if symptom == symp['name']:
      return symp

def getLifeStyle(clinical_case, life):
  return clinical_case['life_style']

def getAntecedent(clinical_case, antec):
  return

def all_equal(prods):
  max = prods[0].prob()
  for prod in prods:
    if prod.prob() != max:
      return False
  return True

def get_max_prob(prods):
  """
    get production rule with the maximum probability.
    ex: A -> B[0.2] | CD[0.5] | E[0.3]: selects A -> CD, and returns [C, D] where C and D non terminals
  """
  if len(prods) == 1: # ex: A -> B[1.0]. No need to select max
    return list(prods[0].rhs()) # return list of elements on rhs
  elif all_equal(prods): # ex: A -> B[0.5] | C[0.5]. Choose at random to diversify words choosen, to broaden language
    index = randint(0, len(prods)-1)
    return list(prods[index].rhs())
  else: # A -> B[0.2] | CD[0.5] | E[0.3]
    max = 0
    maxIndex = 0
    index = 0
    for prod in prods:
      if prod.prob() > max:
        max = prod.prob()
        maxIndex = index

      index += 1
    return list(prods[maxIndex].rhs())

def get_word(grammar, non_terminal):
  """
    recursive algorithm: parses through the grammar (derivation tree) till we reach the leaves.
    At each level, we choose the leaf with the greatest probability.
  """
  prods = grammar.productions(lhs=non_terminal)
  prod = get_max_prob(prods) # list

  phrase = []
  for pr in prod:
    if type(pr) == str:
      if pr != '':
        phrase.append(pr)
    else:
      phrase.append(get_word(grammar, non_terminal=pr))
  
  return ' '.join(phrase)


def remove_spaces(word):
  """
    to principally remove unnecessary spaces when empty string from grammar is returned and spaces before full stop.
  """
  while word.find(' .') != -1:
    word = word.replace(' .', '.')
  
  word = word.replace('  ', ' ')

  return word


def max_similar(keywords, word):
  max = 0
  maxWord = ''
  for key in keywords:
    sq = similar(key, word)
    if sq > max:
      max = sq
      maxWord = key
  
  return maxWord

def symptom_phrase_grammar(entities_ner, symptom_obj):
  """
    We create a grammar with varying probabilities considering the entities present in entities_ner
  """

  symptom_grammar = """
    S -> B1 B2 B3 B4 B5 B6 B7 [1.0]

    A1 -> 'I have' [0.3333] | 'Yes Doctor, I have' [0.3333] | 'I feel' [0.3333]
    A2 -> 'about' [0.25] | 'for about' [0.25] | 'around' [0.25] | 'nearly' [0.25]
    A3 -> 'since' [0.2] | 'for the past' [0.2] | 'for' [0.2] | 'for nearly' [0.2] | 'for about' [0.2]
    A4 -> '. The intensity is' [0.5] | ', and the intensity is quite' [0.5]
    A5 -> 'and it is triggered by' [0.25] | '. It is stimulated when I do' [0.25] | '. It starts when' [0.25] | '. The triggering activity is' [0.25]
    A6 -> '. The pain is located at' [0.25] | ' and the pain is located at' [0.25] | '. The pain is situated at' [0.25] | ' and the pain is positioned at' [0.25]
    A7 -> ". It's evolution is" [0.5] | '. The progress of the symptom is' [0.5]

  """

  if 'SYMPTOM' in entities_ner:
    symptom_grammar = symptom_grammar + "B1 -> A1 '" + symptom_obj['name'] + "' [1.0]\n"

  if 'FREQUENCY' in entities_ner:
    symptom_grammar = symptom_grammar + "B2 -> A2 '" + symptom_obj['frequency'] + "' [1.0] | [0.0]\n"
  else:
    symptom_grammar = symptom_grammar + "B2 -> A2 '" + "none" + "' [0.0] | [1.0]\n"

  if 'DURATION' in entities_ner:
    symptom_grammar = symptom_grammar + "B3 -> A3 '" + symptom_obj['duration'] + "' [1.0] | [0.0]\n"
  else:
    symptom_grammar = symptom_grammar + "B3 -> A3 '" + "none" + "' [0.0] | [1.0]\n"

  if 'INTENSITY' in entities_ner:
    symptom_grammar = symptom_grammar + "B4 -> A4 '" + symptom_obj['degree'] + "' [1.0] | [0.0]\n"
  else:
    symptom_grammar = symptom_grammar + "B4 -> A4 '" + "none" + "' [0.0] | [1.0]\n"

  if 'TRIGGER' in entities_ner:
    symptom_grammar = symptom_grammar + "B5 -> A5 '" + symptom_obj['triggering_activity'] + "' [1.0] | [0.0]\n"
  else:
    symptom_grammar = symptom_grammar + "B5 -> A5 '" + "none" + "' [0.0] | [1.0]\n"

  if 'LOCALISATION' in entities_ner:
    symptom_grammar = symptom_grammar + "B6 -> A6 '" + symptom_obj['localisation'] + "' [1.0] | [0.0]\n"
  else:
    symptom_grammar = symptom_grammar + "B6 -> A6 '" + "none" + "' [0.0] | [1.0]\n"

  if 'EVOLUTION' in entities_ner:
    symptom_grammar = symptom_grammar + "B7 -> A7 '" + symptom_obj['evolution'] + "' [1.0] | [0.0]\n"
  else:
    symptom_grammar = symptom_grammar + "B7 -> A7 '" + "none" + "' [0.0] | [1.0]\n"

  grammar = PCFG.fromstring(symptom_grammar)  # we construct the grammar

  S = grammar.start() # axiom of the grammar
  word = get_word(grammar=grammar, non_terminal=S) # parses grammar through left-most derivation to create the word
  return remove_spaces(word)

similar('smoke', 'smoking')
max_similar(['Cigarette', 'Tramol', 'Marijuana', 'Canabis', 'Medecine', 'Smoke'], 'mari')

def lifestyle_phrase_grammar(entities_ner, life_obj):
  """
    We create a grammar with varying probabilities considering the entities present in entities_ner
  """

  if 'MOSQUITO NET' in entities_ner:
    if life_obj['mosquito'] == True:
      return 'Yes Doctor, I sleep under a mosquito net', True
    else:
      return "No Doctor, I don't sleep under a mosquito net", True
  
  elif 'PET' in entities_ner:
    if life_obj['pet_company'] == "":
      return "No Doctor, I don't have any pet", False
    else:
      pets = life_obj['pet_company'].replace(';', ', ')
      return "I have a " + pets, True
  
  elif 'WATER' in entities_ner:
    return 'I usually drink ' + life_obj['water_quality'], True
  
  elif 'ALCOHOL' in entities_ner:
    gram = """
      S -> A1 A2 A3 [1.0]

      A1 -> 'Yes doctor, I consume alcohol' [0.25] | 'I drink alcohol' [0.25] | 'I usually drink alcohol' [0.25] | 'I take acohol, doctor' [0.25]

      B2 -> 'about' [0.25] | 'for about' [0.25] | 'around' [0.25] | 'nearly' [0.25]
      B3 -> 'since' [0.2] | 'for the past' [0.2] | 'for' [0.2] | 'for nearly' [0.2] | 'for about' [0.2]
    """
    alcohol = None
    for life in life_obj['addiction']:
      if life['name'] in ['Alcohol', 'Beer']:
        alcohol = life
        break
    if alcohol == None:
      return "No doctor, I don't consume any alcohol", False
    
    if 'FREQUENCY' in entities_ner:
      gram = gram + "A2 -> B2 '" + alcohol['frequency'] + "' [1.0] | [0.0]\n"
    else:
      gram = gram + "A2 -> B2 '" + "none" + "' [0.0] | [1.0]\n"

    if 'DURATION' in entities_ner:
      gram = gram + "A3 -> B3 '" + alcohol['duration'] + "' [1.0] | [0.0]\n"
    else:
      gram = gram + "A3 -> B3 '" + "none" + "' [0.0] | [1.0]\n"
    
    grammar = PCFG.fromstring(gram)  # we construct the grammar

    S = grammar.start() # axiom of the grammar
    word = get_word(grammar=grammar, non_terminal=S) # parses grammar through left-most derivation to create the word
    return word, True
  
  
  elif 'DRUG' in entities_ner:
    gram = """
      S -> A1 A2 A3 [1.0]

      B1 -> 'Yes doctor, I consume' [0.25] | 'I take' [0.25] | 'I smoke' [0.25] | 'I usually consume' [0.25]
      B2 -> 'about' [0.25] | 'for about' [0.25] | 'around' [0.25] | 'nearly' [0.25]
      B3 -> 'since' [0.2] | 'for the past' [0.2] | 'for' [0.2] | 'for nearly' [0.2] | 'for about' [0.2]
    """
    drugs = []
    for life in life_obj['addiction']:
      if life['name'] in ['Cigarette', 'Tramol', 'Marijuana', 'Canabis', 'Medecine',]:
        drugs.append(life)

    if len(drugs) == 0:
      return "No doctor, I don't smoke anything.", False
    
    drug = None
    drug_name = max_similar(['Cigarette', 'Tramol', 'Marijuana', 'Canabis', 'Medecine', 'Smoke'], entities_ner['DRUG'])
    if drug_name == 'Smoke':
      all_drugs = ""
      for dr in drugs:
        all_drugs = all_drugs + dr['name'] + ", "
      return 'Yes doctor, I consume ' + all_drugs, True
    
    else:
      for dr in drugs:
        if dr['name'] == drug_name:
          drug = dr
          break

    if drugs == None:
      return "I don't smoke " + drug_name + " doctor", False
    
    gram = gram + "A1 -> B1 '" + drug['name'] + "' [1.0] | [0.0]\n"
    if 'FREQUENCY' in entities_ner:
      gram = gram + "A2 -> B2 '" + drug['frequency'] + "' [1.0] | [0.0]\n"
    else:
      gram = gram + "A2 -> B2 '" + "none" + "' [0.0] | [1.0]\n"

    if 'DURATION' in entities_ner:
      gram = gram + "A3 -> B3 '" + drug['duration'] + "' [1.0] | [0.0]\n"
    else:
      gram = gram + "A3 -> B3 '" + "none" + "' [0.0] | [1.0]\n"
    
    grammar = PCFG.fromstring(gram)  # we construct the grammar

    S = grammar.start() # axiom of the grammar
    word = get_word(grammar=grammar, non_terminal=S) # parses grammar through left-most derivation to create the word
    return word, True
  
  elif 'SPORT' in entities_ner:
    sports = life_obj['physical_activity']
    if len(sports) == 0:
      return "I don't do any sport of physical activity, doctor", False
    all_sports = ""
    for sp in sports:
      all_sports = all_sports + sp['name'] + " about " + sp['frequency'] + ", "
    return "I play " + all_sports, True
  
  elif 'TRAVEL' in entities_ner:
    gram = """
      S -> A1 A2 A3 [1.0]

      B1 -> 'Yes doctor, I traveled to' [0.25] | 'I traveled to' [0.25] | 'I made a trip to' [0.25] | 'I went to' [0.25]
      B2 -> 'about' [0.25] | 'for about' [0.25] | 'around' [0.25] | 'nearly' [0.25]
      B3 -> 'since' [0.5] | 'about' [0.5]
    """
    travels = life_obj['travel']
    
    if len(travels) == 0:
      return "No doctor, I didn't travel recently", False
    
    travel = travels[0]
    gram = gram + "A1 -> B1 '" + travel['location'] + "' [1.0] | [0.0]\n"
    if 'FREQUENCY' in entities_ner:
      gram = gram + "A2 -> B2 '" + travel['frequency'] + "' [1.0] | [0.0]\n"
    else:
      gram = gram + "A2 -> B2 '" + "none" + "' [0.0] | [1.0]\n"

    if 'DURATION' in entities_ner:
      gram = gram + "A3 -> B3 '" + travel['duration'] + " ago.' [1.0] | [0.0]\n"
    else:
      gram = gram + "A3 -> B3 '" + "none" + "' [0.0] | [1.0]\n"
    
    grammar = PCFG.fromstring(gram)  # we construct the grammar

    S = grammar.start() # axiom of the grammar
    word = get_word(grammar=grammar, non_terminal=S) # parses grammar through left-most derivation to create the word
    return word, True

def antecedent_phrase_grammar(entities_ner, antecedent_obj):
  """
    We create a grammar with varying probabilities considering the entities present in entities_ner
  """

  if 'ANTECEDENT' in entities_ner:
    if antecedent_obj['family_antecedents']:
      return 'Yes Doctor, my family history includes ' + antecedent_obj['family_antecedents'], True
    else:
      return "No Doctor, I don't have family history", False
  
  elif 'PREGNANCY' in entities_ner:
    if len(antecedent_obj['obstetrical_antecedent']) == 0:
      return "No Doctor, I don't have obstetrical antecedent", False
    else:
      return "I already had " + antecedent_obj['obstetrical_antecedent'][0]['nb_pregnancy'] + " grossesse(s) " + "and the last one was the " + antecedent_obj['obstetrical_antecedent'][0]['date_of_last_pregnancy'], True
  
  elif 'ALLERGY' in entities_ner:
    gram = """
      S -> A1 A2 [1.0]

      A1 -> 'Yes doctor,' [1.0]
      B1 -> 'it manifests itself by' [0.5] | 'it manifested itself by' [0.5]
    """
    allergies = {}
    manifestations = []
    triggers = []

    for allergy in antecedent_obj['allergy']:
      manifestations.append(allergy['manifestation'])
      if allergy['trigger']:
        triggers.append(allergy['manifestation'] + " which is triggered by " + allergy['trigger'])
      else:
        triggers.append(allergy['manifestation'] + " I don't know the trigger activity")
    
    allergies["manifestation"] = ", ".join(manifestations)
    allergies["trigger"] = ", ".join(triggers)

    if 'TRIGGER' in entities_ner:
      gram = gram + "A2 -> B1 '" + allergies['trigger'] + "' [1.0] | [0.0]\n"
    else:
      if allergies['manifestation']:
        gram = gram + "A2 -> B1 '" + allergies['manifestation'] + "' [1.0] | [0.0]\n"
      else:
        return "No doctor, I don't have any allergy", False
    
    grammar = PCFG.fromstring(gram)  # we construct the grammar

    S = grammar.start() # axiom of the grammar
    word = get_word(grammar=grammar, non_terminal=S) # parses grammar through left-most derivation to create the word
    return word, True
  
  elif 'SURGERY' in entities_ner:
    gram = """
      S -> A1 A2 [1.0]

      A1 -> 'Yes doctor,' [1.0]
      B1 -> 'I have done' [0.3333] | 'I did' [0.3333] | "I've done" [0.3333]
    """
    surgeries = []

    for surg in antecedent_obj['surgery']:
      if surg['date']:
        surgeries.append(surg['name'] + " on the " + datetime.strptime(surg['date'], '%Y-%m-%d').date().strftime('%B %Y'))
      else:
        surgeries.append(surg['name'] + " i don't remember the date")
    
    surgery = ", ".join(surgeries)

    if surgery :
      gram = gram + "A2 -> B1 '" + surgery + "' [1.0] | [0.0]\n"
    else:
      word = "No doctor, I did not do any surgery."
    
    grammar = PCFG.fromstring(gram)  # we construct the grammar

    S = grammar.start() # axiom of the grammar
    word = get_word(grammar=grammar, non_terminal=S) # parses grammar through left-most derivation to create the word
    return word

def generate_text(category, clinical_case=None, response=None, symptom_entities = None, life_style_entities = None, antecedent_entities = None):
  # clinical_case = get_clinical_case('ae538715-2b8a-4680-bfb8-a4ccebf4b988')
  global list_symptoms
  if category == 'salutation':
    return 'Good ' + getPeriod() + ' Doctor', True
  elif category == 'initial_problem':
    return "I don't feel well doctor. " + clinical_case['initial_problem'], True
  elif category == 'repetition':
    return 'repetition', True
  elif category == 'life_style':
    if life_style_entities != None:
      return lifestyle_phrase_grammar(entities_ner=life_style_entities, life_obj=clinical_case['life_style'][0])
    return
  elif category == 'antecedent':
    if antecedent_entities != None:
      return antecedent_phrase_grammar(entities_ner=antecedent_entities, antecedent_obj=clinical_case['medical_antecedent'][0])
    return
  elif category == 'symptoms':
    if symptom_entities != None:
      for symp in list_symptoms:
        if similar(symptom_entities['SYMPTOM'], symp) > 0.8:
          s = symp
          break
      symptoms = getSymptoms(clinical_case, s)
      if symptoms == None:
        return "No doctor, I don't have " + symptom_entities['SYMPTOM'], s, False
      else:
        return symptom_phrase_grammar(entities_ner=symptom_entities, symptom_obj=symptoms), s, True
    return

@api_view(['POST'])
def response(request):
  start = time.time()
  body = json.loads(request.body)

  text = body["question"]
  id = body["clinical_case"]

  symp = None
  

  clinical_case = clinicalCase(request=request, id_clinical_case=id)

  cl = classifyText(text)

  print(cl)
  if cl == "Symptoms":
    if text in ["Hello", "hello"]:
      res = "False classification :-("
    else:
      try:
        symptom_entities = getkeySymptom(text)["data"]
        res, symp, in_case = generate_text('symptoms', clinical_case=clinical_case[0], symptom_entities=symptom_entities)
      except:
        result = {
          "status": False,
          "response": "I don't quite understand, can you repeat",
          "class": cl
        }
        end0 = time.time()
        print("Temps " + str(end0 - start))
        return Response(result, status = status.HTTP_500_INTERNAL_SERVER_ERROR)
  elif cl == "Life Style":
    try:
      life_style_entities = getkeyLifeStyle(text)["data"]
      res, in_case = generate_text('life_style', clinical_case=clinical_case[0], life_style_entities=life_style_entities)
    except:
      result = {
          "status": False,
          "response": "I don't quite understand, can you repeat",
          "class": cl
        }
      end1 = time.time()
      print("Temps " + str(end1 - start))
      return Response(result, status = status.HTTP_500_INTERNAL_SERVER_ERROR)
  elif cl == "Antecedent":
    try:
      antecedent_entities = getkeyAntecedent(text)["data"]
      res, in_case = generate_text('antecedent', clinical_case=clinical_case[0], antecedent_entities=antecedent_entities)
    except:
      result = {
          "status": False,
          "response": "I don't quite understand, can you repeat",
          "class": cl
        }
      end2 = time.time()
      print("Temps " + str(end2 - start))
      return Response(result, status = status.HTTP_500_INTERNAL_SERVER_ERROR)
    
  elif cl == "Salutation":
    res, in_case = generate_text('salutation')
  elif cl == "Initial Problem":
    res, in_case = generate_text('initial_problem', clinical_case=clinical_case[0])
  elif cl == "Repetition":
    if "response" in body:
      resp = body["response"]
      if resp:
        res, in_case = generate_text('repetition',response=resp)
      else:
        result = {
          "status": False,
          "message": "The field response not be empty"
        }
        end3 = time.time()
        print("Temps " + str(end3 - start))
        return Response(result, status = status.HTTP_404_NOT_FOUND)
    else:
      result = {
        "status": False,
        "message": "No field response"
      }
      end4 = time.time()
      print("Temps " + str(end4 - start))
      return Response(result, status = status.HTTP_404_NOT_FOUND)
  try:
    result = {
      "status": in_case,
      "response": res,
      "class": cl,
      "symptom": symp if symp else None
    }
  except:
    result = {
          "status": False,
          "response": "I don't quite understand, can you repeat",
          "class": cl
        }
    end5 = time.time()
    print("Temps " + str(end5 - start))
    return Response(result, status = status.HTTP_500_INTERNAL_SERVER_ERROR)
  
  end6 = time.time()
  print("Temps " + str(end6 - start))
  return Response(result, status.HTTP_200_OK)

def checkDifficulty(rate):
  if 0 <= rate <= 0.4:
    return "EASY"
  elif 0.5 <= rate <= 0.7:
    return "MEDIUM"
  else:
    return "HARD"

def rappel(dict_date, dict_niveau_connaissance):
  global working_memory_capacity
  rap = {}
  name = ""
  date2 = datetime.now().date()
  for key in dict_date.keys():
    date1 = datetime.strptime(dict_date[key][0:10], '%Y-%m-%d').date()
    number_days = (date2 - date1).days
    rap[key] = Decimal(Decimal(dict_niveau_connaissance[key]) * exp(-1 * (number_days / working_memory_capacity)))

  rap = Convert1(sorted(rap.items(), key=lambda x: x[1], reverse=True), {})
  name = list(rap.keys())[-1]
  return name

@api_view(["POST"])
def selectClinicalCase(request):
  body = json.loads(request.body)
  context = {
        'request': request,
    }

  if body == None:
    result = {
      "status": "FAILURE",
      "message": "random, learner, system required"
    }
    return Response(result, status.HTTP_204_NO_CONTENT)
  if 'random' not in body:
    result = {
      "status": "FAILURE",
      "message": "random required"
    }
    return Response(result, status.HTTP_204_NO_CONTENT)
  if 'learner' not in body:
    result = {
      "status": "FAILURE",
      "message": "learner required"
    }
    return Response(result, status.HTTP_204_NO_CONTENT)
  
  system_name = ""
  disease_name = ""
  systems, diseases, details = getRating(request=request, pk=body["learner"])

  systems_date, details_date = getDate(request=request, pk=body["learner"])

  if body["random"]:

    for key in list(systems.keys()):
      if systems[key] == "0.000":
        system_name = key
        dis = details[system_name]
        for key1 in dis.keys():
          if dis[key1] == "0.000":
            disease_name = key1
            case = clinicalCase(request, id_clinical_case="all", system=system_name, diseas=disease_name, diff="EASY")
            if case:
              break
            else:
              disease_name = ""
              continue
        if case :
          break
        else:
          system_name = ""
          continue
    
    if case:
      result = {
        "clinical_case":case if case else None,
      }

      return Response(result, status=status.HTTP_200_OK)
    else:
      system_name = rappel(dict_date=systems_date, dict_niveau_connaissance=systems)
      dis = details[system_name]
      for key in dis.keys():
        if dis[key] == "0.000":
          disease_name = key
          case = clinicalCase(request, id_clinical_case="all", system=system_name, diseas=disease_name, diff="EASY")
          if case:
            break
          else:
            continue
      if case:
        result = {
          "clinical_case":case if case else None,
        }

        return Response(result, status=status.HTTP_200_OK)
      else:
        disease_name = rappel(dict_date=details_date[system_name], dict_niveau_connaissance=details[system_name])
        case = clinicalCase(request, id_clinical_case="all", diseas=disease_name, diff=checkDifficulty(details[system_name][disease_name]))
  else:
    if 'system' not in body:
      result = {
        "status": "FAILURE",
        "message": "system required"
      }
      return Response(result, status.HTTP_204_NO_CONTENT)
    
    dis = details[body["system"]]
    for key in dis.keys():
      if dis[key] == "0.000":
        disease_name = key
        case = clinicalCase(request, id_clinical_case="all", system=system_name, diseas=disease_name, diff="EASY")
        if case:
          break
        else:
          continue
    if case:
      result = {
        "clinical_case":case if case else None,
      }

      return Response(result, status=status.HTTP_200_OK)
    else:
      disease_name = rappel(dict_date=details_date[system_name], dict_niveau_connaissance=details[system_name])
      case = clinicalCase(request, id_clinical_case="all", diseas=disease_name, diff=checkDifficulty(details[system_name][disease_name]))

  result = {
    "clinical_case":case if case else None,
  }

  return Response(result, status=status.HTTP_200_OK)

@api_view(['POST'])
def marking(request):
  body = json.loads(request.body)
  context = {
        'request': request,
    }

  if body == None:
    result = {
      "status": "FAILURE",
      "message": "random, learner, system required"
    }
    return Response(result, status.HTTP_204_NO_CONTENT)
  if 'symptoms' not in body:
    result = {
      "status": "FAILURE",
      "message": "symptoms required"
    }
    return Response(result, status.HTTP_204_NO_CONTENT)
  if 'learner' not in body:
    result = {
      "status": "FAILURE",
      "message": "learner required"
    }
    return Response(result, status.HTTP_204_NO_CONTENT)
  if 'clinical_case' not in body:
    result = {
      "status": "FAILURE",
      "message": "clinical_case required"
    }
    return Response(result, status.HTTP_204_NO_CONTENT)
  if 'procedure' not in body:
    result = {
      "status": "FAILURE",
      "message": "procedure required"
    }
    return Response(result, status.HTTP_204_NO_CONTENT)
  if 'final_diagnostic' not in body:
    result = {
      "status": "FAILURE",
      "message": "final_diagnostic required"
    }
    return Response(result, status.HTTP_204_NO_CONTENT)
  
  if 'exams' not in body:
    result = {
      "status": "FAILURE",
      "message": "exams required"
    }
    return Response(result, status.HTTP_204_NO_CONTENT)
  
  try: 
    learner = LeanerPhysicianSerializer(LeanerPhysician.objects.get(pk=body["learner"]), many=False, context=context).data
  except LeanerPhysician.DoesNotExist: 
    return JsonResponse({'message': 'The learner does not exist'}, status=status.HTTP_404_NOT_FOUND)
  
  try: 
    clinical_case = ClinicalCaseSerializer(CC.objects.get(pk=body["clinical_case"]), many=False, context=context).data
  except CC.DoesNotExist: 
    return JsonResponse({'message': 'The clinical case does not exist'}, status=status.HTTP_404_NOT_FOUND)
  
  symps_case = []
  symps = {}
  exam = {}
  exam_case = []

  note_symps = 0
  note_exam = 0

  clinical = clinicalCase(request, id_clinical_case=body["clinical_case"])[0]

  for s in clinical["symptom"]:
    symps_case.append(s["name"])
  
  for e in clinical["exam"]:
    exam_case.append(e["name"])
  
  symps_case = set(symps_case)
  exam_case = set(exam_case)
  bsymp = set(body["symptoms"])

  symps_utiles = getSymptom1(clinical["final_diagnosis"]) - symps_case

  len_indis = len(bsymp & symps_case)
  len_utile = len((bsymp - symps_case) & symps_utiles)
  len_nefaste = len((bsymp - symps_case) - symps_utiles)

  note_symps = ((2 * len_indis) + len_utile - len_nefaste)/((2 * len(symps_case)) + len(symps_utiles))

  if len(exam_case) == 0 :
    if len(body["exams"]) == 0:
      note_exam = 1
  else:
    if len(body["exams"]) == 0:
      note_exam = 0
    else:
      note_exam = ((2 * len(set(body["exams"]) & exam_case)) - len(set(body["exams"]) - exam_case)) / (2 * len(exam_case))
      if note_exam < 0:
        note_exam = 0
  note = (0.4 * note_symps) + (0.2 * note_exam)
  if body["final_diagnostic"]:
    note += 0.2
  if body["procedure"]:
    note += 0.2
  
  for sy in bsymp:
    if sy in symps_case:
      symps[sy] = 0
    elif sy in symps_utiles:
      symps[sy] = 1
    else:
      symps[sy] = 2
    
  for ex in body["exams"]:
    if ex in exam_case:
      exam[ex] = 0
    else:
      exam[ex] = 1

  result = {
    "note":note,
    "symptoms":symps,
    "exams":exam
  }
  return Response(result, status=status.HTTP_200_OK)

@api_view(['POST'])
def verifySymptom(request):
  body = json.loads(request.body)
  context = {
        'request': request,
    }

  if body == None:
    result = {
      "status": "FAILURE",
      "message": "random, learner, system required"
    }
    return Response(result, status.HTTP_204_NO_CONTENT)
  if 'symptoms' not in body:
    result = {
      "status": "FAILURE",
      "message": "symptoms required"
    }
    return Response(result, status.HTTP_204_NO_CONTENT)
  if 'clinical_case' not in body:
    result = {
      "status": "FAILURE",
      "message": "clinical_case required"
    }
    return Response(result, status.HTTP_204_NO_CONTENT)
  
  try: 
    clinical_case = ClinicalCaseSerializer(CC.objects.get(pk=body["clinical_case"]), many=False, context=context).data
  except CC.DoesNotExist: 
    return JsonResponse({'message': 'The clinical case does not exist'}, status=status.HTTP_404_NOT_FOUND)
  
  symps_case = []
  symps = {}
  clinical = clinicalCase(request, id_clinical_case=body["clinical_case"])[0]

  for s in clinical["symptom"]:
    symps_case.append(s["name"])
  
  symps_case = set(symps_case)
  bsymp = set(body["symptoms"])

  symps_utiles = getSymptom1(clinical["final_diagnosis"]) - symps_case

  for sy in bsymp:
    if sy in symps_case:
      symps[sy] = 0
    elif sy in symps_utiles:
      symps[sy] = 1
    else:
      symps[sy] = 2
  
  return Response(symps, status=status.HTTP_200_OK)

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
    return JsonResponse(result, status=status.HTTP_200_OK)


def root(request):
    return redirect('/api')