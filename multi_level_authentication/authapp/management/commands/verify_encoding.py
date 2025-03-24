# multi_level_authentication/authapp/management/commands/verify_encoding.py

from django.core.management.base import BaseCommand
from authapp.models import BiometricData, User
import numpy as np
import face_recognition

class Command(BaseCommand):
    help = 'Verify face encodings in the database'

    def handle(self, *args, **options):
        self.stdout.write('Starting face encoding verification...')
        
        try:
            # Get all biometric data
            bios = BiometricData.objects.all()
            
            if not bios.exists():
                self.stdout.write(self.style.WARNING('No biometric data found in database'))
                return
                
            for bio in bios:
                try:
                    self.stdout.write(f"\nChecking user: {bio.user.username}")
                    
                    # Get the face encoding
                    face_data = bio.face_encoding
                    
                    if not face_data:
                        self.stdout.write(self.style.WARNING(f"- No face encoding found"))
                        continue
                        
                    # Convert binary to numpy array
                    face_array = np.frombuffer(face_data, dtype=np.float64)
                    
                    # Print details
                    self.stdout.write(f"- Raw binary size: {len(face_data)} bytes")
                    self.stdout.write(f"- Array shape: {face_array.shape}")
                    self.stdout.write(f"- Array type: {face_array.dtype}")
                    self.stdout.write(f"- First 5 values: {face_array[:5]}")
                    
                    # Verify the shape
                    if face_array.shape[0] == 128:
                        self.stdout.write(self.style.SUCCESS("- Valid 128-dimensional encoding"))
                    else:
                        self.stdout.write(self.style.ERROR(f"- Invalid shape: {face_array.shape[0]} (should be 128)"))
                    
                    # Check for NaN or infinite values
                    if np.isnan(face_array).any():
                        self.stdout.write(self.style.ERROR("- Contains NaN values"))
                    if np.isinf(face_array).any():
                        self.stdout.write(self.style.ERROR("- Contains infinite values"))
                    
                    # Print gesture sequence
                    self.stdout.write(f"- Gesture sequence: {bio.gesture_sequence}")
                    
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f"Error processing {bio.user.username}: {str(e)}"))
                    
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error: {str(e)}"))