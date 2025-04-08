from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    pattern_image = models.BinaryField(null=True, blank=True)
    pattern = models.TextField(null=True, blank=True)
    pattern_position_1 = models.IntegerField(null=True, blank=True)
    pattern_position_2 = models.IntegerField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.user.username}'s profile"


class AuthenticationSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    level_one_complete = models.BooleanField(default=False)
    level_two_complete = models.BooleanField(default=False)
    level_three_complete = models.BooleanField(default=False)
    session_key = models.CharField(max_length=100)
    timestamp = models.DateTimeField(auto_now_add=True)



    def __str__(self):
        return self.user.username

class BiometricData(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    face_encoding = models.BinaryField()  # Store face encoding data
    gesture_sequence = models.JSONField(null=True, blank=True)  # Store gesture sequence
    registration_date = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
       return f"Biometric Data for {self.user.username}"
    
    