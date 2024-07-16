from django.contrib.auth.models import User
from rest_framework import serializers
from .models import Substation, Marker


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["id", "username", "email", "first_name", "last_name"]


class MarkerSerializer(serializers.ModelSerializer):
    created_by = serializers.SerializerMethodField()
    updated_by = serializers.SerializerMethodField()

    class Meta:
        model = Marker
        fields = [
            "id",
            "label",
            "latitude",
            "longitude",
            "attributes",
            "substation",
            "created_by",
            "updated_by",
        ]

    def get_created_by(self, obj):
        return obj.substation.created_by.username if obj.substation.created_by else None

    def get_updated_by(self, obj):
        return obj.substation.updated_by.username if obj.substation.updated_by else None


class SubstationSerializer(serializers.ModelSerializer):
    markers = MarkerSerializer(many=True, read_only=True)

    class Meta:
        model = Substation
        fields = [
            "ss_id",
            "created_at",
            "updated_at",
            "created_by",
            "updated_by",
            "markers",
        ]


class MarkerDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = Marker
        fields = ["label", "latitude", "longitude", "attributes"]


class BulkMarkerUpdateSerializer(serializers.Serializer):
    substation_id = serializers.IntegerField()
    created_by = serializers.CharField()
    updated_by = serializers.CharField()
    markers = MarkerDataSerializer(many=True)
