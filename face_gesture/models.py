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
    """Session tracking for multi-level authentication"""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    session_key = models.CharField(max_length=40)
    level_one_complete = models.BooleanField(default=False) 
    level_two_complete = models.BooleanField(default=False)
    level_three_complete = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ('user', 'session_key')
    
    def __str__(self):
        return f"Authentication session for {self.user.username}"
    
    @property
    def is_fully_authenticated(self):
        """Check if all authentication levels are complete"""
        return self.level_one_complete and self.level_two_complete and self.level_three_complete