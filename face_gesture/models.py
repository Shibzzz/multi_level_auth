# accounts/models.py
from django.db import models
from django.contrib.auth.models import User

class BiometricProfile(models.Model):
    """Profile for storing biometric data"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='biometric_profile')
    face_registered = models.BooleanField(default=False)
    gesture_registered = models.BooleanField(default=False)
    face_encoding_path = models.CharField(max_length=255, blank=True, null=True)
    gesture_encoding_path = models.CharField(max_length=255, blank=True, null=True)
    
    def __str__(self):
        return f"BiometricProfile for {self.user.username}"
    
    @property
    def get_registration_status(self):
        return {
            'face_registered': self.face_registered,
            'gesture_registered': self.gesture_registered
        }

class UserProfile(models.Model):
    """Profile for storing image pattern authentication data"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='pattern_profile')
    pattern_image = models.BinaryField(null=True)
    pattern = models.TextField(null=True, help_text="JSON string of grid pattern")
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"UserProfile for {self.user.username}"

class AuthenticationSession(models.Model):
    """Session for tracking multi-level authentication status"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='auth_sessions')
    session_key = models.CharField(max_length=255)
    level_one_complete = models.BooleanField(default=False)
    level_two_complete = models.BooleanField(default=False)
    level_three_complete = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"AuthSession for {self.user.username} ({self.session_key})"
    
    @property
    def is_fully_authenticated(self):
        return self.level_one_complete and self.level_two_complete and self.level_three_complete