from django.urls import path, include
from django.contrib import admin
from rest_framework.routers import DefaultRouter
from core import views
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

# Gis mapping routing urls
gis_router = DefaultRouter()
gis_router.register(r'users', views.UserViewSet)
gis_router.register(r'substations', views.SubstationViewSet)
gis_router.register(r'markers', views.MarkerViewSet)
gis_router.register(r'bulk_update', views.BulkMarkerUpdateViewSet, basename='bulk_update')

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include(gis_router.urls)),
    path('gis/', include(gis_router.urls)),
    path('gis/bulk_update/update_markers/', views.BulkMarkerUpdateViewSet.as_view({'post': 'update_markers'}), name='bulk-update-markers'),
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
]
