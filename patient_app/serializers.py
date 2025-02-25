from .models import *
from rest_framework import serializers
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer

class RegistrationSerializer(serializers.ModelSerializer):
    
    class Meta:
        model = Doctor
        fields = [
            'id','url',
            'name',
            'first_name',
            'phone_number',
            'sex',
            'specialty',
            'year_of_birth',
            'email',
            'username',
            'password',
            'created_at',
            'deleted_at',
            'updated_at'
        ]
    
    
    def validate(self,args):
        email = args.get('email', None)
        username = args.get('username', None)
        if Doctor.objects.filter(email=email).exists():
            raise serializers.ValidationError({'email' : ('email already exists')})
        if Doctor.objects.filter(username=username).exists():
            raise serializers.ValidationError({'username' : ('usernale already exists')})
       
        return super().validate(args)
    
    def create(self, validated_data):
        return Doctor.objects.create(**validated_data)
        #return User.objects.create_user(**validated_data)

# class TokenObtainSerializer(TokenObtainPairSerializer):
#     def validate(self, attrs):

#         username=attrs.get('username', None)
#         password=attrs.get('password', None)
#         queryset = Doctor.objects.get(username=username, password=password)
#         context = {
#             'request': None
#         }
#         print(attrs)
#         data = super().validate(attrs)
#         refresh = self.get_token(self.user)

#         data["refresh"] = str(refresh)
#         data["access"] = str(refresh.access_token)
#         serializer = DoctorSerializer(queryset, context=context)

#         try:
#             que1 = LeanerPhysician.objects.get(username=username, password=password)
#             serializer = LeanerPhysicianSerializer(que1, context=context)
#             data["type"] = "LEARNER"

#         except:

#             print("Ce n'est pas un apprenant")

#             try:
#                 que2 = ExpertPhysician.objects.get(username=username, password=password)
#                 serializer = ExpertPhysicianSerializer(que2, context=context)
#                 data["type"] = "EXPERT"
#             except:
#                 print("ce n'est pas un expert")
#                 data["type"] = "ADMIN"
            
#         data["Doctor"] = serializer.data

#         return data

def check_password(user, password):
    if user.password == password:
        return True
    else:
        return False


class TokenObtainSerializer(TokenObtainPairSerializer):
    def validate(self, attrs):
        username = attrs.get('username', None)
        password = attrs.get('password', None)

        # Vérification des credentials
        user = Doctor.objects.filter(username=username).first()
        
        if user is None:
            #raise serializers.ValidationError({"error": "User doesn't exist", "success": False})
            return {"error": "User does not exist"}
        
        if not check_password(user, password):
            #raise serializers.ValidationError({"error": "Incorrect Password", "success": False})
            return {"error": "Incorrect Password"}

        context = {
            'request': None
        }
        data = super().validate(attrs)
        refresh = self.get_token(user)
        serializer = None

        data["refresh"] = str(refresh)
        data["access"] = str(refresh.access_token)

        # Récupération des informations supplémentaires sur l'utilisateur
        try:
            que1 = LeanerPhysician.objects.get(username=username)
            serializer = LeanerPhysicianSerializer(que1, context=context)
            data["type"] = "LEARNER"
        except LeanerPhysician.DoesNotExist:
            try:
                que2 = ExpertPhysician.objects.get(username=username)
                serializer = ExpertPhysicianSerializer(que2, context=context)
                data["type"] = "EXPERT"
            except ExpertPhysician.DoesNotExist:

                data["type"] = "ADMIN"

        data["Doctor"] = serializer.data

        return data

class DoctorSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Doctor
        fields = [
            'id','url',
            'name',
            'first_name',
            'phone_number',
            'sex',
            'specialty',
            'year_of_birth',
            'email',
            'username',
            'password',
            'password',
            'created_at',
            'deleted_at',
            'updated_at'
            'is_admin'
        ]

class LeanerPhysicianSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = LeanerPhysician
        fields = [
            'id','url',
            'name',
            'first_name',
            'phone_number',
            'sex',
            'specialty',
            'year_of_birth',
            'email',
            'username',
            'password',
            'experience',
            'knowledge_level',
            'created_at',
            'deleted_at',
            'updated_at'
        ]

class ExpertPhysicianSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = ExpertPhysician
        fields = [
            'id','url',
            'name',
            'first_name',
            'phone_number',
            'sex',
            'specialty',
            'year_of_birth',
            'email',
            'username',
            'password',
            'grade',
            'created_at',
            'deleted_at',
            'updated_at'
        ]

class LogSerializer(serializers.HyperlinkedModelSerializer):

   

    class Meta:
        model = Log
        fields = [
            'id','url',
            'operation',
            'table',
            'doctor',
            'created_at',
            'deleted_at',
            'updated_at'
        ]

class VirtualPatientSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = VirtualPatient
        fields = [
            'id','url',
            'sex',
            'civil_status',
            'min_age',
            'max_age',
            'weight',
            'modele_3D',
            'created_at',
            'deleted_at',
            'updated_at'
        ]

class ClinicalCaseSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = ClinicalCase
        fields = [
            'id','url',
            'initial_problem',
            'difficulty',
            'final_diagnosis',
            'system',
            'specialty',
            'created_at',
            'deleted_at',
            'updated_at'
        ]

class VirtualCaseSerializer(serializers.HyperlinkedModelSerializer):


    class Meta:
        model = VirtualCase
        fields = [
            'id','url',
            'virtual_patient',
            'clinical_case',
            'created_at',
            'deleted_at',
            'updated_at'
        ]


class EvaluationSerializer(serializers.HyperlinkedModelSerializer):


    class Meta:
        model = Evaluation
        fields = [
            'id','url',
            'type',
            'duration',
            'mark',
            'symptom',
            'final_diagnosis',
            'learner_physician',
            'virtual_case',
            'created_at',
            'deleted_at',
            'updated_at'
        ]

class FeedbackSerializer(serializers.HyperlinkedModelSerializer):


    class Meta:
        model = Feedback
        fields = [
            'id','url',
            'comment',
            'evaluation',
            'expert_physician',
            'created_at',
            'deleted_at',
            'updated_at'
        ]

class DiagnosisSerializer(serializers.HyperlinkedModelSerializer):


    class Meta:
        model = Diagnosis
        fields = [
            'id','url',
            'type',
            'name',
            'result',
            'verdict',
            'evaluation',
            'created_at',
            'deleted_at',
            'updated_at'
        ]

class HypothesisSerializer(serializers.HyperlinkedModelSerializer):


    class Meta:
        model = Hypothesis
        fields = [
            'id','url',
            'symptoms',
            'diseases',
            'reason',
            'threshold',
            'learner_physician',
            'evaluation',
            'created_at',
            'deleted_at',
            'updated_at'
        ]

class QuestionSerializer(serializers.HyperlinkedModelSerializer):


    class Meta:
        model = Question
        fields = [
            'id','url',
            'question',
            'answer',
            'status',
            'learner_physician',
            'evaluation',
            'created_at',
            'deleted_at',
            'updated_at'
        ]

class PersonalInfoSerializer(serializers.HyperlinkedModelSerializer):


    class Meta:
        model = PersonalInfo
        fields = [
            'id','url',
            'sex',
            'age',
            'civil_status',
            'profession',
            'nb_child',
            'blood_group',
            'clinical_case',
            'created_at',
            'deleted_at',
            'updated_at'
        ]

class TreatmentInProgressSerializer(serializers.HyperlinkedModelSerializer):


    class Meta:
        model = TreatmentInProgress
        fields = [
            'id','url',
            'name',
            'administration_mode',
            'duration',
            'observation',
            'efficiency',
            'clinical_case',
            'created_at',
            'deleted_at',
            'updated_at'
        ]

class MediaSerializer(serializers.HyperlinkedModelSerializer):

    class Meta:
        model = Media
        fields = [
            'id','url',
            'name',
            'type',
            'file',
            'created_at',
            'deleted_at',
            'updated_at'
        ]

class PhysicalDiagnosisSerializer(serializers.HyperlinkedModelSerializer):

    class Meta:
        model = PhysicalDiagnosis
        fields = [
            'id','url',
            'physical_diagnosis',
            'result',
            'file',
            'clinical_case',
            'created_at',
            'deleted_at',
            'updated_at'
        ]

class ExamSerializer(serializers.HyperlinkedModelSerializer):

    class Meta:
        model = Exam
        fields = [
            'id','url',
            'name',
            'anatomy',
            'result',
            'verdict',
            'file',
            'clinical_case',
            'created_at',
            'deleted_at',
            'updated_at'
        ]


class TypeParameterSerializer(serializers.HyperlinkedModelSerializer):

    class Meta:
        model = TypeParameter
        fields = [
            'id',
            'url',
            'name',
            'unit',
            'created_at',
            'deleted_at',
            'updated_at'
        ]

class MedicalParameterSerializer(serializers.HyperlinkedModelSerializer):

    class Meta:
        model = MedicalParameter
        fields = [
            'id','url',
            'value',
            'comment',
            'type_parameter',
            'clinical_case',
            'created_at',
            'deleted_at',
            'updated_at'
        ]

class LifeStyleSerializer(serializers.HyperlinkedModelSerializer):

    class Meta:
        model = LifeStyle
        fields = [
            'id','url',
            'water_quality',
            'mosquito',
            'pet_company',
            'clinical_case',
            'created_at',
            'deleted_at',
            'updated_at'
        ]

class PhysicalActivitySerializer(serializers.HyperlinkedModelSerializer):

    class Meta:
        model = PhysicalActivity
        fields = [
            'id','url',
            'name',
            'frequency',
            'life_style',
            'created_at',
            'deleted_at',
            'updated_at'
        ]
    
class AddictionSerializer(serializers.HyperlinkedModelSerializer):

    class Meta:
        model = Addiction
        fields = [
            'id','url',
            'name',
            'frequency',
            'duration',
            'life_style',
            'created_at',
            'deleted_at',
            'updated_at'
        ]

class TravelSerializer(serializers.HyperlinkedModelSerializer):

    class Meta:
        model = Travel
        fields = [
            'id','url',
            'location',
            'frequency',
            'duration',
            'life_style',
            'created_at',
            'deleted_at',
            'updated_at'
        ]

class SymptomSerializer(serializers.HyperlinkedModelSerializer):

    class Meta:
        model = Symptom
        fields = [
            'id','url',
            'name',
            'localisation',
            'frequency',
            'duration',
            'evolution',
            'triggering_activity',
            'degree',
            'clinical_case',
            'created_at',
            'deleted_at',
            'updated_at'
        ]


"""class ConceptSerializer(serializers.HyperlinkedModelSerializer):

    class Meta:
        model = Concept
        fields = [
            'id','url',
            'name',
            'clinical_case',
            'created_at',
            'deleted_at',
            'updated_at'
        ]"""

class MedicalAntecedentSerializer(serializers.HyperlinkedModelSerializer):

    class Meta:
        model = MedicalAntecedent
        fields = [
            'id','url',
            'family_antecedents',
            'clinical_case',
            'created_at',
            'deleted_at',
            'updated_at'
        ]

class ObstetricalAntecedentSerializer(serializers.HyperlinkedModelSerializer):

    class Meta:
        model = ObstetricalAntecedent
        fields = [
            'id','url',
            'nb_pregnancy',
            'date_of_last_pregnancy',
            'medical_antecedent',
            'created_at',
            'deleted_at',
            'updated_at'
        ]

class SurgerySerializer(serializers.HyperlinkedModelSerializer):

    class Meta:
        model = Surgery
        fields = [
            'id','url',
            'name',
            'date',
            'medical_antecedent',
            'created_at',
            'deleted_at',
            'updated_at'
        ]

class AllergySerializer(serializers.HyperlinkedModelSerializer):

    class Meta:
        model = Allergy
        fields = [
            'id','url',
            'manifestation',
            'trigger',
            'medical_antecedent',
            'created_at',
            'deleted_at',
            'updated_at'
        ]

class DiseaseSerializer(serializers.HyperlinkedModelSerializer):

    class Meta:
        model = Disease
        fields = [
            'id','url',
            'name',
            'start_time',
            'end_time',
            'observation',
            'medical_antecedent',
            'created_at',
            'deleted_at',
            'updated_at'
        ]


class TreatmentSerializer(serializers.HyperlinkedModelSerializer):

    class Meta:
        model = Treatment
        fields = [
            'id','url',
            'name',
            'duration',
            'posology',
            'disease',
            'created_at',
            'deleted_at',
            'updated_at'
        ]