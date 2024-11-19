from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.contrib.auth.models import User
from .models import Substation, Marker
from .serializers import (
    UserSerializer,
    SubstationSerializer,
    MarkerSerializer,
    BulkMarkerUpdateSerializer,
)
from django.core.cache import cache


# GIS Mapping Viewsets
class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [IsAuthenticated]

    @action(detail=False, methods=["get"], permission_classes=[IsAuthenticated])
    def me(self, request):
        serializer = self.get_serializer(request.user)
        return Response(status=status.HTTP_200_OK, data=serializer.data)


class SubstationViewSet(viewsets.ModelViewSet):
    queryset = Substation.objects.all()
    serializer_class = SubstationSerializer
    permission_classes = [IsAuthenticated]

    @action(detail=True, methods=["get"])
    def markers(self, request, pk=None):
        substation = self.get_object()
        cache_key = f"substation_{pk}_markers"
        markers = cache.get(cache_key)

        if markers is None:
            markers = substation.markers.all()
            cache.set(cache_key, markers, timeout=300)  # Cache for 5 minutes

        serializer = MarkerSerializer(markers, many=True)
        return Response(serializer.data)


class MarkerViewSet(viewsets.ModelViewSet):
    queryset = Marker.objects.all()
    serializer_class = MarkerSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        queryset = super().get_queryset()
        substation_id = self.request.query_params.get("substation")
        if substation_id:
            cache_key = f"substation_{substation_id}_markers"
            markers = cache.get(cache_key)

            if markers is None:
                queryset = queryset.filter(substation__ss_id=substation_id)
                cache.set(cache_key, queryset, timeout=300)  # Cache for 5 minutes
            else:
                queryset = markers
        return queryset


class BulkMarkerUpdateViewSet(viewsets.ViewSet):
    permission_classes = [IsAuthenticated]

    @action(detail=False, methods=["post"])
    def update_markers(self, request):
        print(request.data)
        serializer = BulkMarkerUpdateSerializer(data=request.data)
        if serializer.is_valid():
            substation_id = serializer.validated_data["substation_id"]
            created_by_username = serializer.validated_data["created_by"]
            updated_by_username = serializer.validated_data["updated_by"]
            markers_data = serializer.validated_data["markers"]

            try:
                created_by = User.objects.get(username=created_by_username)
            except User.DoesNotExist:
                return Response(
                    {"error": f"User with username {created_by_username} not found"},
                    status=status.HTTP_404_NOT_FOUND,
                )

            try:
                updated_by = User.objects.get(username=updated_by_username)
            except User.DoesNotExist:
                return Response(
                    {"error": f"User with username {updated_by_username} not found"},
                    status=status.HTTP_404_NOT_FOUND,
                )

            substation, created = Substation.objects.get_or_create(
                ss_id=substation_id,
                defaults={"created_by": created_by, "updated_by": updated_by},
            )

            if not created:
                substation.updated_by = updated_by
                substation.save()
                Marker.objects.filter(substation=substation).delete()

            new_markers = []
            for marker_data in markers_data:
                new_marker = Marker(
                    substation=substation,
                    label=marker_data["label"],
                    latitude=marker_data["latitude"],
                    longitude=marker_data["longitude"],
                    attributes=marker_data["attributes"],
                )
                new_markers.append(new_marker)

            Marker.objects.bulk_create(new_markers)
            return Response({"status": "markers updated"})
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)