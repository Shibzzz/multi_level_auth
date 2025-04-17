from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import get_user_model
from .face_gesture.utils.face_detection import capture_face
from .face_gesture.utils.face_recog import FaceVerifier
from .face_gesture.utils.gesture_detect import GestureDetector
import json
import base64
import numpy as np
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

# Face Registration and Verification Views
def face_registration_page(request):
    """Render the face registration page."""
    return render(request, 'face_gesture/face_registration.html', {
        'user_id': request.user.id if request.user.is_authenticated else None
    })

@login_required
def face_verification_page(request):
    """Render the face verification page."""
    return render(request, 'face_gesture/face_verification.html', {
        'user_id': request.user.id
    })

@csrf_exempt
def register_face(request, user_id):
    """Handle face registration process."""
    if request.method == 'POST':
        try:
            success = capture_face(str(user_id))
            
            if success:
                return JsonResponse({
                    'status': 'success',
                    'message': 'Face registration completed successfully!'
                })
            else:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Face registration failed. Please try again.'
                })
                
        except Exception as e:
            logger.error(f"Error during registration: {str(e)}")
            return JsonResponse({
                'status': 'error',
                'message': f'An error occurred during registration: {str(e)}'
            })
    
    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request method'
    })

@csrf_exempt
@login_required
def verify_face(request, user_id):
    """Handle face verification process."""
    if str(request.user.id) != str(user_id):
        return JsonResponse({
            'status': 'error',
            'message': 'Unauthorized access'
        }, status=403)

    if request.method == 'POST':
        try:
            verifier = FaceVerifier()
            
            if not verifier.load_known_face(str(user_id)):
                return JsonResponse({
                    'status': 'error',
                    'message': 'No face registration found. Please register your face first.'
                })
            
            success = verifier.verify_face(str(user_id))
            
            if success:
                return JsonResponse({
                    'status': 'success',
                    'message': 'Face verification completed successfully!'
                })
            else:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Face verification failed. Please try again.'
                })
                
        except Exception as e:
            logger.error(f"Error during verification: {str(e)}")
            return JsonResponse({
                'status': 'error',
                'message': f'An error occurred during verification: {str(e)}'
            })
    
    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request method'
    })

# Gesture Registration and Detection Views
def gesture_registration_page(request):
    """Render the gesture registration page."""
    return render(request, 'face_gesture/gesture_register.html', {
        'user_id': request.user.id if request.user.is_authenticated else None
    })

def gesture_verification_page(request):
    """Render the gesture verification page."""
    return render(request, 'face_gesture/gesture_verify.html', {
        'user_id': request.user.id if request.user.is_authenticated else None
    })

@csrf_exempt
def register_gesture(request):
    """Handle gesture registration process."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            username = data.get('username')
            gesture_images = data.get('images', [])
            
            if not username or not gesture_images:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Missing username or gesture images'
                })

            # Initialize gesture detector
            detector = GestureDetector()
            
            # Process and save each gesture image
            for idx, img_data in enumerate(gesture_images):
                # Convert base64 to image
                img_data = img_data.split(',')[1]  # Remove data URL prefix
                img_bytes = base64.b64decode(img_data)
                img = Image.open(io.BytesIO(img_bytes))
                
                # Save gesture data
                success = detector.save_gesture(username, img, idx)
                if not success:
                    return JsonResponse({
                        'status': 'error',
                        'message': f'Failed to save gesture {idx+1}'
                    })

            return JsonResponse({
                'status': 'success',
                'message': 'Gesture registration completed successfully!'
            })

        except Exception as e:
            logger.error(f"Error during gesture registration: {str(e)}")
            return JsonResponse({
                'status': 'error',
                'message': f'An error occurred: {str(e)}'
            })

    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request method'
    })

@csrf_exempt
def verify_gesture(request):
    """Handle gesture verification process."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            username = data.get('username')
            gesture_image = data.get('image')
            
            if not username or not gesture_image:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Missing username or gesture image'
                })

            # Convert base64 to image
            img_data = gesture_image.split(',')[1]  # Remove data URL prefix
            img_bytes = base64.b64decode(img_data)
            img = Image.open(io.BytesIO(img_bytes))
            
            # Initialize gesture detector and verify
            detector = GestureDetector()
            success, confidence = detector.verify_gesture(username, img)
            
            if success:
                return JsonResponse({
                    'status': 'success',
                    'message': f'Gesture verified successfully! (Confidence: {confidence:.2f})'
                })
            else:
                return JsonResponse({
                    'status': 'error',
                    'message': f'Gesture verification failed. (Confidence: {confidence:.2f})'
                })

        except Exception as e:
            logger.error(f"Error during gesture verification: {str(e)}")
            return JsonResponse({
                'status': 'error',
                'message': f'An error occurred: {str(e)}'
            })

    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request method'
    }) 

