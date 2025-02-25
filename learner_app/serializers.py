from .models import *
from rest_framework import serializers

class SymptomSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Symptom
        fields = [
            'id',
            'url',
            'name',
            'created_at',
            'deleted_at',
            'updated_at'
        ]

class DiseaseSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Disease
        fields = [
            'id',
            'url',
            'name',
            'created_at',
            'deleted_at',
            'updated_at'
        ]

class WeightSymptomDiseaseSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = WeightSymptomDisease
        fields = [
            'id',
            'url',
            'weight',
            'symptom',
            'disease',
            'created_at',
            'deleted_at',
            'updated_at'
        ]

class SystemSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = System
        fields = [
            'id',
            'url',
            'name',
            'created_at',
            'deleted_at',
            'updated_at'
        ]

class WeightDiseaseSystemSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = WeightDiseaseSystem
        fields = [
            'id',
            'url',
            'weight',
            'system',
            'disease',
            'created_at',
            'deleted_at',
            'updated_at'
        ]

class RatingDiseaseSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = RatingDisease
        fields = [
            'id',
            'url',
            'rating',
            'learner',
            'disease',
            'system',
            'created_at',
            'deleted_at',
            'updated_at'
        ]

class RatingSystemSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = RatingSystem
        fields = [
            'id',
            'url',
            'rating',
            'learner',
            'system',
            'created_at',
            'deleted_at',
            'updated_at'
        ]