from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import User
from .models import Substation, Marker

# Unregister the default User admin
admin.site.unregister(User)

# Register the customized User admin
@admin.register(User)
class CustomUserAdmin(UserAdmin):
    list_display = ('username', 'email', 'first_name', 'last_name', 'is_staff')
    search_fields = ('username', 'email', 'first_name', 'last_name')
    list_filter = ('is_staff', 'is_superuser', 'is_active', 'groups')
    ordering = ('username',)

# Register the Substation model
@admin.register(Substation)
class SubstationAdmin(admin.ModelAdmin):
    list_display = ('ss_id', 'created_at', 'updated_at', 'created_by', 'updated_by')
    search_fields = ('ss_id', 'created_by__username', 'updated_by__username')
    list_filter = ('created_at', 'updated_at')
    readonly_fields = ('created_at', 'updated_at')

# Register the Marker model
@admin.register(Marker)
class MarkerAdmin(admin.ModelAdmin):
    list_display = ('label', 'latitude', 'longitude', 'substation')
    search_fields = ('label', 'substation__ss_id')
    list_filter = ('substation',)
