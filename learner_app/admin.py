from django.contrib import admin
from .models import *

# Register your models here.
admin.site.register(Symptom)
admin.site.register(Disease)
admin.site.register(WeightSymptomDisease)
admin.site.register(System)
admin.site.register(WeightDiseaseSystem)
admin.site.register(RatingDisease)
admin.site.register(RatingSystem)