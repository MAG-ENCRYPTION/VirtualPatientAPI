from random import choices
from django.db import models
from django.contrib.auth.models import BaseUserManager
import uuid

# Create your models here.

class UserManager(BaseUserManager):
    def create_user(self, email, name = None, first_name = None, username = None, password=None, is_active=True, is_staff=False, is_admin=False):
        if not email:
            raise ValueError("User must have an email")
        if not password:
            raise ValueError("User must have a password")
        user_obj = self.model(
            username = username,
            first_name = first_name,
            last_name = name,
            email = email
        )
        user_obj.set_password(password)
        user_obj.staff = is_staff
        user_obj.admin = is_admin
        user_obj.active = is_active
        #user_obj.save(using=self._db)
        user_obj.save()
        return user_obj
    
    def create_staffuser(self, email, name = None, first_name = None, username = None, password=None):
        user = self.create_user(
            email,
            name = name,
            first_name = first_name,
            username = username, 
            password=password, 
            is_staff=True
        )
        return user
    
    def create_superuser(self, email = None, name = None, first_name = None, username = None, password=None):
        user = self.create_user(
            email,
            name = name,
            first_name = first_name,
            username = username, 
            password=password,  
            is_staff=True,
            is_admin=True
        )
        return user

class Doctor(models.Model):
    """
        doctor model
    """
    SEX = (
        ('M', 'Male'),
        ('F', 'Female'),
    )

    SPECIALTY = (
        ('Generalist', 'Generalist'),
        ('Neurologist', 'Neurologist'),
        ('Gynecologist', 'Gynecologist'),
        ('Pediatrician', 'Pediatrician'),
        ('Dentist', 'Dentist'),
        ('Ophthalmologist', 'Ophthalmologist'),
    )

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=50, null=False)
    first_name = models.CharField(max_length=50, null=False)
    phone_number = models.CharField(max_length=20, default='+237 ', unique=True)
    sex = models.CharField(choices=SEX, max_length=50)
    specialty = models.CharField(choices=SPECIALTY, max_length=50)
    year_of_birth = models.DateField(null=False)
    email = models.EmailField(max_length=50, unique=True, null=False)
    username = models.CharField(max_length=50, null=False)
    password = models.CharField(max_length=50, null=False)
    created_at = models.DateTimeField(null=False, auto_now_add=True)
    deleted_at = models.DateTimeField(null=False, auto_now_add=True)
    updated_at = models.DateTimeField(null=False, auto_now=True)
    is_admin = models.BooleanField(default=False)

    objects = UserManager()

    def __str__(self):
        return self.name + " " + self.first_name + " " + str(self.year_of_birth)


class LeanerPhysician(Doctor):
    EXPERIENCE = (
        ("Beginner", "Beginner"),
        ("Intermediate", "Intermediate"),
        ("Expert", "Expert")
    )

    experience = models.CharField(max_length=20, choices=EXPERIENCE)
    knowledge_level = models.DecimalField(default=0.0, null=True, decimal_places=2, max_digits=6)

class ExpertPhysician(Doctor):
    GRADE = (
        ('GP', 'Generalist Physician'),
        ('SP', 'Specialist Physician'),
        ('Prof', 'Professor')
    )
    grade = models.CharField(max_length=20, choices=GRADE, null=False)



class Log(models.Model):

    OPERATION = (
        ('UPDATE', 'UPDATE'),
        ('CREATE', 'CREATE'),
        ('DELETE', 'DELETE')
    )

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    operation = models.CharField(max_length=20, choices=OPERATION)
    table = models.CharField(max_length=50)
    doctor = models.ForeignKey(Doctor, on_delete=models.SET_NULL, null=True)
    created_at = models.DateTimeField(null=False, auto_now_add=True)
    deleted_at = models.DateTimeField(null=False, auto_now_add=True)
    updated_at = models.DateTimeField(null=False, auto_now=True)

class VirtualPatient(models.Model):

    SEX = (
        ('M', 'Male'),
        ('F', 'Female'),
    )

    CIVIL_STATUS = (
        ('SINGLE', 'SINGLE'),
        ('MARRIED', 'MARRIED'),
        ('DIVORCED', 'DIVORCED')
    )

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    sex = models.CharField(choices=SEX, max_length=50)
    civil_status = models.CharField(choices=CIVIL_STATUS, max_length=50)
    min_age = models.IntegerField(default=0, null=False)
    max_age = models.IntegerField(default=0, null=False)
    weight = models.DecimalField(default=0.0, null=False, decimal_places=2, max_digits=6)
    modele_3D = models.CharField(max_length=50, null=False)
    created_at = models.DateTimeField(null=False, auto_now_add=True)
    deleted_at = models.DateTimeField(null=False, auto_now_add=True)
    updated_at = models.DateTimeField(null=False, auto_now=True)

class ClinicalCase(models.Model):

    DIFFICULTE = (
        ('EASY', 'EASY'),
        ('MEDIUM', 'MEDIUM'),
        ('HARD', 'HARD')
    )

    SYSTEM = (
        ('RESPIRATORY SYSTEM', 'RESPIRATORY SYSTEM'),
        ('CARDIOVASCULAR SYSTEM', 'CARDIOVASCULAR SYSTEM'),
        ('MUSCULAR SYSTEM', 'MUSCULAR SYSTEM'),
        ('NERVOUS SYSTEM', 'NERVOUS SYSTEM'),
        ('DIGESTIVE SYSTEM', 'DIGESTIVE SYSTEM'),
        ('URINARY SYSTEM', 'URINARY SYSTEM'),
        ('ENDOCRINE SYSTEM', 'ENDOCRINE SYSTEM'),
        ('LYMPHATIC SYSTEM', 'LYMPHATIC SYSTEM'),
        ('REPRODUCTIVE SYSTEM', 'REPRODUCTIVE SYSTEM')
    )

    SPECIALTY = (
        ('Generalist', 'Generalist'),
        ('Neurologist', 'Neurologist'),
        ('Gynecologist', 'Gynecologist'),
        ('Pediatrician', 'Pediatrician'),
        ('Dentist','Dentist'),
        ('Ophthalmologist', 'Ophthalmologist'),
    )

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    initial_problem =  models.TextField(null=False)
    difficulty = models.CharField(choices=DIFFICULTE, max_length=50)
    final_diagnosis = models.CharField(null=False, max_length=100)
    system = models.CharField(max_length=150, null=False)
    specialty = models.CharField(choices=SPECIALTY, max_length=50)
    created_at = models.DateTimeField(null=False, auto_now_add=True)
    deleted_at = models.DateTimeField(null=False, auto_now_add=True)
    updated_at = models.DateTimeField(null=False, auto_now=True)

class VirtualCase(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    virtual_patient = models.ForeignKey(VirtualPatient, on_delete=models.SET_NULL, null=True)
    clinical_case = models.ForeignKey(ClinicalCase, on_delete=models.SET_NULL, null=True)
    created_at = models.DateTimeField(null=False, auto_now_add=True)
    deleted_at = models.DateTimeField(null=False, auto_now_add=True)
    updated_at = models.DateTimeField(null=False, auto_now=True)


class Evaluation(models.Model):

    TYPE = (
        ('FORMATIF', 'FORMATIF'),
        ('SOMMATIF', 'SOMMATIF')
    )

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    type = models.CharField(max_length=20, choices=TYPE)
    duration = models.IntegerField(default=0, null=False)
    mark = models.DecimalField(default=0.0, null=True, decimal_places=2, max_digits=5)
    symptom = models.CharField(max_length=150, null=True)
    final_diagnosis = models.CharField(max_length=150, null=True)
    learner_physician = models.ForeignKey(LeanerPhysician, on_delete=models.SET_NULL, null=True)
    virtual_case = models.ForeignKey(VirtualCase, on_delete=models.SET_NULL, null=True)
    created_at = models.DateTimeField(null=False, auto_now_add=True)
    deleted_at = models.DateTimeField(null=False, auto_now_add=True)
    updated_at = models.DateTimeField(null=False, auto_now=True)

class Feedback(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    comment = models.TextField(blank=True)
    evaluation = models.ForeignKey(Evaluation, on_delete=models.DO_NOTHING, null=True)
    expert_physician = models.ForeignKey(ExpertPhysician, on_delete=models.CASCADE, null=True)
    created_at = models.DateTimeField(null=False, auto_now_add=True)
    deleted_at = models.DateTimeField(null=False, auto_now_add=True)
    updated_at = models.DateTimeField(null=False, auto_now=True)

class Diagnosis(models.Model):

    TYPE = (
        ('PHYSICAL DIAGNOSIS', 'PHYSICAL DIAGNOSIS'),
        ('EXAM', 'EXAM')
    )

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    type = models.CharField(max_length=20, choices=TYPE)
    name = models.CharField(max_length=50, null=False)
    result = models.CharField(max_length=50, null=True)
    verdict = models.CharField(max_length=50, null=False)
    evaluation = models.ForeignKey(Evaluation, on_delete=models.CASCADE, null=True)
    created_at = models.DateTimeField(null=False, auto_now_add=True)
    deleted_at = models.DateTimeField(null=False, auto_now_add=True)
    updated_at = models.DateTimeField(null=False, auto_now=True)

class Hypothesis(models.Model):

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    symptoms = models.CharField(max_length=150, null=False)
    diseases = models.CharField(max_length=150, null=False)
    reason = models.TextField(blank=False)
    threshold = models.DecimalField(max_digits=30, decimal_places=20)
    learner_physician = models.ForeignKey(LeanerPhysician, on_delete=models.SET_NULL, null=True)
    evaluation = models.ForeignKey(Evaluation, on_delete=models.CASCADE, null=True)
    created_at = models.DateTimeField(null=False, auto_now_add=True)
    deleted_at = models.DateTimeField(null=False, auto_now_add=True)
    updated_at = models.DateTimeField(null=False, auto_now=True)

class Question(models.Model):

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    question = models.TextField(blank=False)
    answer = models.TextField(blank=False)
    status = models.BooleanField(default=False)
    learner_physician = models.ForeignKey(LeanerPhysician, on_delete=models.SET_NULL, null=True)
    evaluation = models.ForeignKey(Evaluation, on_delete=models.CASCADE, null=True)
    created_at = models.DateTimeField(null=False, auto_now_add=True)
    deleted_at = models.DateTimeField(null=False, auto_now_add=True)
    updated_at = models.DateTimeField(null=False, auto_now=True)

# Information du cas clinique
class PersonalInfo(models.Model):
    SEX = (
        ('M', 'Male'),
        ('F', 'Female'),
    )

    CIVIL_STATUS = (
        ('SINGLE', 'SINGLE'),
        ('MARRIED', 'MARRIED'),
        ('DIVORCED', 'DIVORCED')
    )

    BLOOD_GROUP = (
        ('A+', 'A+'),
        ('A-', 'A-'),
        ('B+', 'B+'),
        ('B-', 'B-'),
        ('AB+', 'AB+'),
        ('AB-', 'AB-'),
        ('O-', 'O-'),
        ('O+', 'O+')
    )

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    sex = models.CharField(choices=SEX, max_length=50)
    age = models.IntegerField(null=False)
    civil_status = models.CharField(choices=CIVIL_STATUS, max_length=50)
    profession = models.CharField(max_length=100, null=True)
    nb_child = models.IntegerField(null=True)
    blood_group = models.CharField(choices=BLOOD_GROUP, null=True, max_length=50)
    clinical_case = models.OneToOneField(ClinicalCase, on_delete=models.CASCADE)
    created_at = models.DateTimeField(null=False, auto_now_add=True)
    deleted_at = models.DateTimeField(null=False, auto_now_add=True)
    updated_at = models.DateTimeField(null=False, auto_now=True)

class TreatmentInProgress(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=50, null=False)
    administration_mode = models.CharField(max_length=100)
    duration = models.CharField(max_length=100, null=True)
    observation = models.TextField(blank=True)
    efficiency = models.CharField(max_length=50, null=True)
    clinical_case = models.ForeignKey(ClinicalCase, on_delete=models.CASCADE, null=False)
    created_at = models.DateTimeField(null=False, auto_now_add=True)
    deleted_at = models.DateTimeField(null=False, auto_now_add=True)
    updated_at = models.DateTimeField(null=False, auto_now=True)

class Media(models.Model):
    TYPES = (('EXAM', 'EXAM'),
    ('PHYSICAL DIAGNOSIS', 'PHYSICAL DIAGNOSIS'))
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=150, null=False)
    type = models.CharField(choices=TYPES, max_length=50, null=False)
    file = models.FileField(null=False)
    created_at = models.DateTimeField(null=False, auto_now_add=True)
    deleted_at = models.DateTimeField(null=False, auto_now_add=True)
    updated_at = models.DateTimeField(null=False, auto_now=True)

class PhysicalDiagnosis(models.Model):

    DIAGNOSIS_PHYSICS = (
        ('PALPATION', 'PALPATION'),
        ('OSCULTATION', 'OSCULTATION'),
        ('PERCUTION', 'PERCUTION'),
        ('INSPECTION', 'INSPECTION')
    )

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    physical_diagnosis = models.CharField(max_length=50, choices=DIAGNOSIS_PHYSICS)
    result = models.TextField(blank=False)
    file = models.ForeignKey(Media, on_delete=models.CASCADE, null=True)
    clinical_case = models.ForeignKey(ClinicalCase, on_delete=models.CASCADE, null=False)
    created_at = models.DateTimeField(null=False, auto_now_add=True)
    deleted_at = models.DateTimeField(null=False, auto_now_add=True)
    updated_at = models.DateTimeField(null=False, auto_now=True)

class Exam(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100, null=False)
    anatomy = models.CharField(max_length=50)
    result = models.TextField(blank=False)
    verdict = models.BooleanField(default=False)
    file = models.ForeignKey(Media, on_delete=models.CASCADE, null=True)
    clinical_case = models.ForeignKey(ClinicalCase, on_delete=models.CASCADE, null=False)
    created_at = models.DateTimeField(null=False, auto_now_add=True)
    deleted_at = models.DateTimeField(null=False, auto_now_add=True)
    updated_at = models.DateTimeField(null=False, auto_now=True)


class TypeParameter(models.Model):

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=30)
    unit = models.CharField(max_length=20, blank=True)
    created_at = models.DateTimeField(null=False, auto_now_add=True)
    deleted_at = models.DateTimeField(null=False, auto_now_add=True)
    updated_at = models.DateTimeField(null=False, auto_now=True)

    def __str__(self):
        return self.name + " en " + self.unit
class MedicalParameter(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    value = models.DecimalField(max_digits=10, decimal_places=2)
    comment = models.TextField(blank=True)
    type_parameter = models.ForeignKey(TypeParameter, on_delete=models.CASCADE, null=False)
    clinical_case = models.ForeignKey(ClinicalCase, on_delete=models.CASCADE, null=False)
    created_at = models.DateTimeField(null=False, auto_now_add=True)
    deleted_at = models.DateTimeField(null=False, auto_now_add=True)
    updated_at = models.DateTimeField(null=False, auto_now=True)

    def __str__(self):
        return self.type_parameter.name + " = " + self.value + " " + self.type_parameter.unit

class LifeStyle(models.Model):

    WATER_QUALITY = (
        ('Mineral water', 'Mineral water'),
        ('Tap water', 'Tap water'),
        ('Water from the source', 'Water from the source'),
        ('River water', 'River water'),
        ('Potable water', 'Potable water'),
        ('Well water', 'Well water')
    )
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    water_quality = models.CharField(max_length=50, choices=WATER_QUALITY, null=True)
    mosquito = models.BooleanField(default=False, null=False)
    pet_company = models.CharField(max_length=50, null=True)
    clinical_case = models.ForeignKey(ClinicalCase, on_delete=models.CASCADE, null=False)
    created_at = models.DateTimeField(null=False, auto_now_add=True)
    deleted_at = models.DateTimeField(null=False, auto_now_add=True)
    updated_at = models.DateTimeField(null=False, auto_now=True)

class PhysicalActivity(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=50)
    frequency = models.CharField(max_length=50)
    life_style = models.ForeignKey(LifeStyle, on_delete=models.CASCADE, null=False)
    created_at = models.DateTimeField(null=False, auto_now_add=True)
    deleted_at = models.DateTimeField(null=False, auto_now_add=True)
    updated_at = models.DateTimeField(null=False, auto_now=True)

class Addiction(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=50)
    frequency = models.CharField(max_length=50)
    duration = models.CharField(max_length=50, null=True)
    life_style = models.ForeignKey(LifeStyle, on_delete=models.CASCADE, null=False)
    created_at = models.DateTimeField(null=False, auto_now_add=True)
    deleted_at = models.DateTimeField(null=False, auto_now_add=True)
    updated_at = models.DateTimeField(null=False, auto_now=True)

class Travel(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    location = models.CharField(max_length=50)
    frequency = models.CharField(max_length=50)
    duration = models.CharField(max_length=50, null=True)
    life_style = models.ForeignKey(LifeStyle, on_delete=models.CASCADE, null=False)
    created_at = models.DateTimeField(null=False, auto_now_add=True)
    deleted_at = models.DateTimeField(null=False, auto_now_add=True)
    updated_at = models.DateTimeField(null=False, auto_now=True)

class Symptom(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    localisation = models.CharField(max_length=100)
    frequency = models.CharField(max_length=100)
    duration = models.CharField(max_length=100, null=True)
    evolution = models.CharField(max_length=150)
    triggering_activity = models.CharField(max_length=100, null=True)
    degree = models.CharField(max_length=100)
    clinical_case = models.ForeignKey(ClinicalCase, on_delete=models.CASCADE, null=False)
    created_at = models.DateTimeField(null=False, auto_now_add=True)
    deleted_at = models.DateTimeField(null=False, auto_now_add=True)
    updated_at = models.DateTimeField(null=False, auto_now=True)

"""class Concept(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    clinical_case = models.ForeignKey(ClinicalCase, on_delete=models.CASCADE, null=False)
    created_at = models.DateTimeField(null=False, auto_now_add=True)
    deleted_at = models.DateTimeField(null=False, auto_now_add=True)
    updated_at = models.DateTimeField(null=False, auto_now=True)"""

class MedicalAntecedent(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    family_antecedents = models.CharField(max_length=100)
    clinical_case = models.OneToOneField(ClinicalCase, on_delete=models.CASCADE, null=False)
    created_at = models.DateTimeField(null=False, auto_now_add=True)
    deleted_at = models.DateTimeField(null=False, auto_now_add=True)
    updated_at = models.DateTimeField(null=False, auto_now=True)

class ObstetricalAntecedent(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    nb_pregnancy = models.IntegerField()
    date_of_last_pregnancy = models.DateField(null=True)
    medical_antecedent = models.OneToOneField(MedicalAntecedent, on_delete=models.CASCADE, null=False)
    created_at = models.DateTimeField(null=False, auto_now_add=True)
    deleted_at = models.DateTimeField(null=False, auto_now_add=True)
    updated_at = models.DateTimeField(null=False, auto_now=True)

class Surgery(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    date = models.DateField(null=True)
    medical_antecedent = models.ForeignKey(MedicalAntecedent, on_delete=models.CASCADE, null=False)
    created_at = models.DateTimeField(null=False, auto_now_add=True)
    deleted_at = models.DateTimeField(null=False, auto_now_add=True)
    updated_at = models.DateTimeField(null=False, auto_now=True)

class Allergy(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    manifestation = models.TextField(blank=False)
    trigger = models.CharField(max_length=200, null=True)
    medical_antecedent = models.ForeignKey(MedicalAntecedent, on_delete=models.CASCADE, null=False)
    created_at = models.DateTimeField(null=False, auto_now_add=True)
    deleted_at = models.DateTimeField(null=False, auto_now_add=True)
    updated_at = models.DateTimeField(null=False, auto_now=True)


class Disease(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    start_time = models.DateField(null=True)
    end_time = models.DateField(null=True)
    observation = models.TextField(blank=True)
    medical_antecedent = models.ForeignKey(MedicalAntecedent, on_delete=models.CASCADE, null=False)
    created_at = models.DateTimeField(null=False, auto_now_add=True)
    deleted_at = models.DateTimeField(null=False, auto_now_add=True)
    updated_at = models.DateTimeField(null=False, auto_now=True)

class Treatment(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    duration = models.CharField(max_length=150 ,null=True)
    posology = models.TextField(blank=True)
    disease = models.ForeignKey(Disease, on_delete=models.CASCADE, null=False)
    created_at = models.DateTimeField(null=False, auto_now_add=True)
    deleted_at = models.DateTimeField(null=False, auto_now_add=True)
    updated_at = models.DateTimeField(null=False, auto_now=True)