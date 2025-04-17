# accounts/models.py
from django.db import models
from django.contrib.auth.models import User

class BiometricProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    face_registered = models.BooleanField(default=False)
    gesture_registered = models.BooleanField(default=False)
    face_encoding_path = models.CharField(max_length=255, null=True)
    gesture_encoding_path = models.CharField(max_length=255, null=True)