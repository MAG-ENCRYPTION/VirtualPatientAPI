from django.db import models
import uuid

from patient_app import models as patient_model

# Create your models here.
class Symptom(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(null=False, auto_now_add=True)
    deleted_at = models.DateTimeField(null=False, auto_now_add=True)
    updated_at = models.DateTimeField(null=False, auto_now=True)

    def __str__(self):
        return self.name 

class Disease(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(null=False, auto_now_add=True)
    deleted_at = models.DateTimeField(null=False, auto_now_add=True)
    updated_at = models.DateTimeField(null=False, auto_now=True)

    def __str__(self):
        return self.name 

class WeightSymptomDisease(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    weight = models.DecimalField(max_digits=10, decimal_places=3)
    symptom = models.ForeignKey(Symptom, on_delete=models.DO_NOTHING, null=False)
    disease = models.ForeignKey(Disease, on_delete=models.DO_NOTHING, null=False)
    created_at = models.DateTimeField(null=False, auto_now_add=True)
    deleted_at = models.DateTimeField(null=False, auto_now_add=True)
    updated_at = models.DateTimeField(null=False, auto_now=True)

class System(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(null=False, auto_now_add=True)
    deleted_at = models.DateTimeField(null=False, auto_now_add=True)
    updated_at = models.DateTimeField(null=False, auto_now=True)

    def __str__(self):
        return self.name 

class WeightDiseaseSystem(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    weight = models.DecimalField(max_digits=10, decimal_places=3)
    system = models.ForeignKey(System, on_delete=models.DO_NOTHING, null=False)
    disease = models.ForeignKey(Disease, on_delete=models.DO_NOTHING, null=False)
    created_at = models.DateTimeField(null=False, auto_now_add=True)
    deleted_at = models.DateTimeField(null=False, auto_now_add=True)
    updated_at = models.DateTimeField(null=False, auto_now=True)

class RatingDisease(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    rating = models.DecimalField(max_digits=10, decimal_places=3, default=0.0)
    learner = models.ForeignKey(patient_model.LeanerPhysician, on_delete=models.CASCADE, null=False)
    disease = models.ForeignKey(Disease, on_delete=models.DO_NOTHING, null=False)
    system = models.ForeignKey(System, on_delete=models.DO_NOTHING, null=True)
    created_at = models.DateTimeField(null=False, auto_now_add=True)
    deleted_at = models.DateTimeField(null=False, auto_now_add=True)
    updated_at = models.DateTimeField(null=False, auto_now=True)

class RatingSystem(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    rating = models.DecimalField(max_digits=10, decimal_places=3, default=0.0)
    learner = models.ForeignKey(patient_model.LeanerPhysician, on_delete=models.CASCADE, null=False)
    system = models.ForeignKey(System, on_delete=models.DO_NOTHING, null=False)
    created_at = models.DateTimeField(null=False, auto_now_add=True)
    deleted_at = models.DateTimeField(null=False, auto_now_add=True)
    updated_at = models.DateTimeField(null=False, auto_now=True)