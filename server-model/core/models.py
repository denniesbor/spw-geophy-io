from django.db import models
from django.contrib.auth.models import User

class Substation(models.Model):
    ss_id = models.IntegerField(unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(User, related_name="created_substations", on_delete=models.CASCADE)
    updated_by = models.ForeignKey(User, related_name="updated_substations", on_delete=models.CASCADE, null=True, blank=True)

    def save(self, *args, **kwargs):
        if not self.updated_by:
            self.updated_by = self.created_by
        super().save(*args, **kwargs)

    def __str__(self):
        return str(self.ss_id)
    

# Marker data for the substation
markers = {
    "Transformer": 
        {
            "phase": ['Single Phase', 'Three Phase'], # Only select a single item from each list
            "role":['Distribution', "Transmission"], # Only select a single item from each list
            "transformer_fuel_type": ["Oil", "Dry", "Gas"], # Only select a single item from each list
        },
    "Circuit Breaker":
        {
            "type":["type1", "type2", "type3", 'type4','type5'], # Only select a single item from each list
        },
    "Power Lines": {
        "type": ["intra-site", "extra-site"],
    },
    "Controls": {
        "type": ["Facility", "low voltage", "high voltage", 'switchgear'] # Only select a single item from each list
    },
    "Reactors": {
        "type": ["shunt", "air core"] # either shunt or air core
        },
    "Alt. Energy": {
        "type":["battery", "capacitor", "wind component", "PV System"]
        },
    # Marker for unclassed a user can add name, type and description. Name is mandatory
    "Other": {
        "name": '', 
        "type": "", 
        "description": ""
    }
}

class Marker(models.Model):
    substation = models.ForeignKey(Substation, related_name="markers", on_delete=models.CASCADE)
    label = models.CharField(max_length=100)
    latitude = models.FloatField()
    longitude = models.FloatField()
    attributes = models.JSONField(default=dict, blank=True)  # JSONField to store arbitrary attributes
    def __str__(self):
        return f"Marker {self.label} at ({self.latitude}, {self.longitude})"
