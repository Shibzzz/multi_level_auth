from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.urls import reverse
from .models import AuthenticationSession, BiometricProfile, UserProfile
from .forms import UserRegistrationForm, UserLoginForm
from .utils.face_recog import FaceVerifier
from .utils.gesture_detect import GestureRegistrar
from .utils.gesture_recog import GestureVerifier
import os
import cv2
import numpy as np
from PIL import Image
import io
import json
import random
import logging
import face_recognition
import mediapipe as mp
from io import BytesIO
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import traceback
from datetime import datetime


logger = logging.getLogger(__name__)

def process_frame_data(frame_data):
    """Convert frame data to OpenCV format."""
    try:
        # Convert the uploaded image to a PIL Image
        image = Image.open(io.BytesIO(frame_data))
        
        # Convert PIL image to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return cv_image
    except Exception as e:
        logger.error(f"Error processing frame data: {str(e)}")
        return None

def home(request):
    """Home view to show options for authentication."""
    return render(request, 'face_gesture/home.html')

def register(request):
    """Level one registration using username and password."""
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            # Store the cleaned data in session
            request.session['registration_data'] = {
                'username': form.cleaned_data['username'],
                'email': form.cleaned_data['email'],
                'password1': form.cleaned_data['password1'],
                'password2': form.cleaned_data['password2'],
            }
            
            # Debug log
            logger.info("Registration data stored in session, redirecting to level two")
            
            return redirect('register_level_two')
        else:
            # Log form errors for debugging
            logger.error(f"Form validation errors: {form.errors}")
            messages.error(request, 'Please correct the errors below.')
    else:
        form = UserRegistrationForm()
    
    return render(request, 'face_gesture/register.html', {'form': form})

@csrf_exempt
def register_level_two(request):
    """Level two registration using pattern authentication."""
    # Check if level one registration data exists in session
    if 'registration_data' not in request.session:
        messages.error(request, 'Please complete level one registration first')
        return redirect('register')
    
    if request.method == 'POST':
        try:
            # Debug log request
            logger.info(f"POST request received with content type: {request.content_type}")
            logger.info(f"POST data keys: {list(request.POST.keys())}")
            logger.info(f"FILES keys: {list(request.FILES.keys())}")
            
            # Get the uploaded image
            if 'image' not in request.FILES:
                logger.error("No image file received in request.FILES")
                return JsonResponse({'status': 'error', 'message': 'No image uploaded. Please try again with a valid image.'})
                
            image = request.FILES['image']
            logger.info(f"Received image file: {image.name}, size: {image.size} bytes")
            
            # Get the pattern data from the form
            pattern_str = request.POST.get('pattern')
            if not pattern_str:
                logger.error("No pattern data received in request.POST")
                return JsonResponse({'status': 'error', 'message': 'No pattern data received. Please try again.'})
            
            # Parse the pattern JSON
            try:
                pattern_data = json.loads(pattern_str)
                if not pattern_data:
                    logger.error("Empty pattern data after parsing JSON")
                    return JsonResponse({'status': 'error', 'message': 'Invalid pattern format - empty data'})
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {str(e)}")
                return JsonResponse({'status': 'error', 'message': f'Invalid pattern format: {str(e)}'})
            
            # Extract sequence data from the pattern
            try:
                sequence = pattern_data.get('sequence', [])
                if not sequence:
                    logger.error("No sequence found in pattern data")
                    return JsonResponse({'status': 'error', 'message': 'Invalid pattern format - no sequence data'})
                    
                logger.info(f"Sequence data: {sequence}")
                
                # Verify sequence has all 9 positions and follows the fixed pattern rule
                # (each piece must be in its matching position)
                if len(sequence) != 9:
                    logger.error(f"Invalid sequence length: expected 9, got {len(sequence)}")
                    return JsonResponse({'status': 'error', 'message': 'Please place all 9 pieces in the pattern grid'})
                
                # For each placement, check that blockId matches position
                # (block 1 must be in position 1, etc.)
                correct_placement = True
                for item in sequence:
                    blockId = item.get('blockId')
                    position = item.get('position')
                    
                    if blockId != position:
                        logger.error(f"Invalid placement: block {blockId} in position {position}")
                        correct_placement = False
                        break
                        
                if not correct_placement:
                    return JsonResponse({
                        'status': 'error', 
                        'message': 'Each block must be placed in its matching position (block 1 in position 1, etc.)'
                    })
                    
                # Get placements data for ordering
                placements = pattern_data.get('placements', [])
                
                # Sort placements by time
                sorted_placements = sorted(placements, key=lambda x: int(x.get('placementTime', 0)))
                
                # Create user from registration data
                registration_data = request.session.get('registration_data', {})
                form = UserRegistrationForm(registration_data)
            except KeyError as e:
                logger.error(f"Missing key in pattern data: {str(e)}")
                return JsonResponse({'status': 'error', 'message': f'Invalid pattern format: {str(e)}'})
            
            if form.is_valid():
                try:
                    user = form.save()
                    logger.info(f"User created successfully: {user.username}")
                    
                    # Log the user in
                    login(request, user)
                    logger.info("User logged in successfully")
                    
                    # Ensure session is created and has a key
                    if not request.session.session_key:
                        request.session.create()
                        logger.info(f"Created new session with key: {request.session.session_key}")
                    else:
                        logger.info(f"Using existing session key: {request.session.session_key}")
                except Exception as e:
                    logger.error(f"Error saving user: {str(e)}")
                    return JsonResponse({'status': 'error', 'message': f'Error creating user: {str(e)}'})
                
                # Read and process image
                try:
                    # Open image using PIL
                    try:
                        img = Image.open(image)
                    except Exception as e:
                        logger.error(f"Error opening image: {str(e)}")
                        return JsonResponse({'status': 'error', 'message': f'Error opening image: {str(e)}'})
                    
                    # Resize image to a reasonable size (e.g., 800x800)
                    max_size = (800, 800)
                    img.thumbnail(max_size, Image.LANCZOS)
                    
                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Save to BytesIO
                    img_io = BytesIO()
                    img.save(img_io, format='JPEG', quality=85)
                    img_io.seek(0)
                    
                    # Create pattern object with simplified format
                    pattern_object = {
                        'placements': sorted_placements,
                        'sequence': sequence,
                        'registered_at': datetime.now().isoformat()
                    }
                    
                    # Create and save user profile with processed image data
                    try:
                        user_profile = UserProfile.objects.create(
                            user=user,
                            pattern_image=img_io.getvalue(),  # Store processed binary data
                            pattern=json.dumps(pattern_object)
                        )
                        logger.info("User profile created successfully with pattern")
                    except Exception as e:
                        logger.error(f"Error creating user profile: {str(e)}")
                        return JsonResponse({'status': 'error', 'message': f'Error saving pattern: {str(e)}'})
                    
                    # Create authentication session with session key
                    try:
                        auth_session = AuthenticationSession.objects.create(
                            user=user,
                            level_one_complete=True,
                            level_two_complete=True,
                            session_key=request.session.session_key
                        )
                        logger.info(f"Authentication session created successfully with key: {auth_session.session_key}")
                    except Exception as e:
                        logger.error(f"Error creating authentication session: {str(e)}")
                        # Try to get existing session
                        auth_session = AuthenticationSession.objects.filter(
                            user=user,
                            session_key=request.session.session_key
                        ).first()
                        if auth_session:
                            logger.info("Found existing authentication session")
                            auth_session.level_one_complete = True
                            auth_session.level_two_complete = True
                            auth_session.save()
                        else:
                            logger.error("Failed to create or find authentication session")
                            return JsonResponse({'status': 'error', 'message': 'Failed to create authentication session'})
                    
                    # Get the redirect URL and log it
                    redirect_url = reverse('register_level_three')
                    logger.info(f"Redirecting to: {redirect_url}")
                    
                    return JsonResponse({
                        'status': 'success',
                        'message': 'Pattern successfully registered!',
                        'redirect_url': redirect_url
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing image: {str(e)}")
                    return JsonResponse({'status': 'error', 'message': f'Error processing pattern: {str(e)}'})
            else:
                logger.error(f"Form validation errors: {form.errors}")
                return JsonResponse({'status': 'error', 'message': 'Invalid registration data'})
        except Exception as e:
            logger.exception(f"Unhandled error in register_level_two: {str(e)}")
            return JsonResponse({'status': 'error', 'message': f'An error occurred: {str(e)}'})
    
    # GET request - render the registration form
    return render(request, 'face_gesture/register_pattern_level_two.html')

def register_level_three(request):
    """Level three registration for biometric auth (face and gesture)."""
    # Check if level one and two registration data exists in session
    if not request.user.is_authenticated:
        messages.error(request, 'Please complete level one and two registration first')
        return redirect('register')
    
    return render(request, 'face_gesture/register_face_level_three.html')

def register_face_level_three_api(request):
    """API endpoint for face registration during the registration process (level three)."""
    if not request.method == 'POST':
        return JsonResponse({
            'status': 'error',
            'message': 'Only POST method is allowed'
        }, status=405)
    
    # Check if user is authenticated (completed levels one and two)
    if not request.user.is_authenticated:
        return JsonResponse({
            'status': 'error',
            'message': 'Please complete level one and two registration first'
        }, status=400)

    if 'frame' not in request.FILES:
        return JsonResponse({
            'status': 'error',
            'message': 'No frame data received'
        }, status=400)

    try:
        # Get username from the authenticated user
        username = request.user.username
        
        # Process the frame data
        frame_data = request.FILES['frame'].read()
        frame = process_frame_data(frame_data)
        
        if frame is None:
            return JsonResponse({
                'status': 'error',
                'message': 'Could not process frame data'
            }, status=400)
        
        # Get capture number
        capture_number = int(request.POST.get('capture_number', 1))
        logger.info(f"Processing face capture {capture_number} for user {username}")
        
        # Ensure the user directory exists
        user_dir = os.path.join('faces', username)
        os.makedirs(user_dir, exist_ok=True)
        
        # Save individual capture
        capture_path = os.path.join(user_dir, f"{username}_capture_{capture_number}.jpg")
        cv2.imwrite(capture_path, frame)
        
        # Process the frame for face encoding
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        
        if not face_locations:
            return JsonResponse({
                'status': 'error',
                'message': 'No face detected in frame'
            }, status=400)
        
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        if not face_encodings:
            return JsonResponse({
                'status': 'error',
                'message': 'Could not generate face encoding'
            }, status=400)
        
        # Save individual encoding
        encoding_path = os.path.join(user_dir, f"{username}_encoding_{capture_number}.npy")
        np.save(encoding_path, face_encodings[0])
        
        # If this is the final capture, create average encoding
        if capture_number == 5:
            try:
                # Load all encodings
                encodings = []
                for i in range(1, 6):
                    enc_path = os.path.join(user_dir, f"{username}_encoding_{i}.npy")
                    encodings.append(np.load(enc_path))
                
                # Calculate average encoding
                average_encoding = np.mean(encodings, axis=0)
                
                # Save final encoding
                final_encoding_path = os.path.join(user_dir, f"{username}_encoding.npy")
                np.save(final_encoding_path, average_encoding)
                
                # Store face registration success in session
                request.session['face_registered'] = True
                request.session['face_encoding_path'] = final_encoding_path
                
                # Create or update BiometricProfile for the user
                profile, created = BiometricProfile.objects.get_or_create(user=request.user)
                profile.face_registered = True
                profile.face_encoding_path = final_encoding_path
                profile.save()
                
                # Clean up individual encodings
                for i in range(1, 6):
                    enc_path = os.path.join(user_dir, f"{username}_encoding_{i}.npy")
                    if os.path.exists(enc_path):
                        os.remove(enc_path)
                        
                logger.info(f"Face registration completed for user {username}")
            except Exception as e:
                logger.error(f"Error creating average encoding: {str(e)}")
                return JsonResponse({
                    'status': 'error',
                    'message': 'Error creating final face encoding'
                }, status=500)
        
        return JsonResponse({
            'status': 'success',
            'message': f'Capture {capture_number} successful'
        })
            
    except Exception as e:
        logger.error(f"Error during face registration: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'message': f'Error during face registration: {str(e)}'
        }, status=500)

def register_gesture_level_three(request):
    """Level three gesture registration part."""
    # Check if level one, two and face registration are complete
    if not request.user.is_authenticated:
        messages.error(request, 'Please complete previous registration steps first')
        return redirect('register')
    
    profile = BiometricProfile.objects.filter(user=request.user, face_registered=True).first()
    if not profile:
        messages.error(request, 'Please complete face registration first')
        return redirect('register_level_three')
    
    return render(request, 'face_gesture/register_gesture_level_three.html')

def register_gesture_level_three_api(request):
    """API endpoint for gesture registration during the registration process."""
    if not request.method == 'POST':
        return JsonResponse({
            'status': 'error',
            'message': 'Only POST method is allowed'
        }, status=405)
    
    # Check if user is authenticated and has registered face
    if not request.user.is_authenticated:
        return JsonResponse({
            'status': 'error',
            'message': 'Please complete previous registration steps first'
        }, status=400)
    
    profile = BiometricProfile.objects.filter(user=request.user, face_registered=True).first()
    if not profile:
        return JsonResponse({
            'status': 'error',
            'message': 'Please complete face registration first'
        }, status=400)

    try:
        # Get username from the authenticated user
        username = request.user.username
        
        # Register user's gesture
        registrar = GestureRegistrar()
        success, save_file, consistency = registrar.register_gesture(username)
        
        if success:
            # Update biometric profile
            profile.gesture_registered = True
            profile.gesture_encoding_path = save_file
            profile.save()
            
            logger.info(f"Gesture registration completed for user {username}")
            
            # Create full authentication session
            auth_session = AuthenticationSession.objects.filter(
                user=request.user,
                session_key=request.session.session_key
            ).first()
            
            if auth_session:
                auth_session.level_three_complete = True
                auth_session.save()
            
            return JsonResponse({
                'status': 'success',
                'message': f'Gesture registered successfully! Consistency: {consistency:.2%}',
                'redirect_url': reverse('register_complete')
            })
        else:
            return JsonResponse({
                'status': 'error',
                'message': 'Gesture registration failed. Please try again.'
            }, status=400)
            
    except Exception as e:
        logger.error(f"Error during gesture registration: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'message': f'Error during registration: {str(e)}'
        }, status=500)

def register_complete(request):
    """Final registration step - confirm successful registration."""
    # Check if all registration levels are complete
    if not request.user.is_authenticated:
        messages.error(request, 'Please complete all registration steps first')
        return redirect('register')
    
    auth_session = AuthenticationSession.objects.filter(
        user=request.user,
        session_key=request.session.session_key,
        level_one_complete=True,
        level_two_complete=True,
        level_three_complete=True
    ).first()
    
    if not auth_session:
        messages.error(request, 'Please complete all registration steps first')
        return redirect('register')
    
    # Clear registration data from session
    for key in ['registration_data', 'face_registered', 'gesture_registered', 
                'face_encoding_path', 'gesture_encoding_path']:
        if key in request.session:
            del request.session[key]
    
    logger.info(f"User {request.user.username} successfully registered with all authentication levels")
    
    return render(request, 'face_gesture/register_complete.html', {
        'username': request.user.username
    })

def register_cancel(request):
    """Cancel the registration process."""
    # Clear registration data from session
    for key in ['registration_data', 'face_registered', 'gesture_registered', 
                'face_encoding_path', 'gesture_encoding_path']:
        if key in request.session:
            del request.session[key]
    
    # If user was created but registration not completed, delete the user
    if request.user.is_authenticated:
        try:
            user = request.user
            logout(request)
            user.delete()
            logger.info(f"Deleted incomplete user registration: {user.username}")
        except Exception as e:
            logger.error(f"Error deleting incomplete user: {str(e)}")
    
    messages.info(request, 'Registration has been cancelled')
    return redirect('home')

def generate_random_grid_images():
    """Generate random images for the grid"""
    # List of possible image paths - adding more options
    image_options = [
        'static/images/grid1.jpg',
        'static/images/grid2.jpg',
        'static/images/grid3.jpg',
        'static/images/grid4.jpg',
        'static/images/grid5.jpg',
        'static/images/grid6.jpg',
        'static/images/grid7.jpg',
        'static/images/grid8.jpg',
        'static/images/grid9.jpg',
        'static/images/grid10.jpg',
        'static/images/grid11.jpg',
        'static/images/grid12.jpg'
    ]
    
    # Randomly select and shuffle 9 images
    selected_images = random.sample(image_options, 9)
    random.shuffle(selected_images)
    
    return selected_images

def level_one(request):
    """Level one authentication using username and password."""
    if request.method == 'POST':
        form = UserLoginForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            
            # Log the user in
            login(request, user)
            
            # Ensure session is created
            if not request.session.session_key:
                request.session.create()
            
            # Create or update authentication session
            auth_session, created = AuthenticationSession.objects.get_or_create(
                user=user,
                session_key=request.session.session_key,
                defaults={'level_one_complete': True}
            )
            
            if not created:
                auth_session.level_one_complete = True
                auth_session.save()
            
            logger.info(f"User {user.username} successfully authenticated level one")
            return redirect('level_two')
        else:
            logger.warning(f"Failed login attempt with form errors: {form.errors}")
    else:
        form = UserLoginForm()
    
    return render(request, 'face_gesture/login.html', {'form': form})

@login_required
def level_two(request):
    """Level two authentication using pattern recognition."""
    # Check if level one is complete
    auth_session = AuthenticationSession.objects.filter(
        user=request.user,
        session_key=request.session.session_key
    ).first()
    
    if not auth_session or not auth_session.level_one_complete:
        messages.error(request, 'You must complete level one authentication first')
        return redirect('level_one')
    
    # Check if user has a pattern registered in UserProfile
    try:
        user_profile = UserProfile.objects.get(user=request.user)
        if not user_profile.pattern:
            messages.warning(request, 'You need to complete the registration process first')
            return redirect('home')
        
        # Redirect to pattern authentication page
        return redirect('pattern_authentication')
    except UserProfile.DoesNotExist:
        # If no UserProfile exists, the user hasn't completed level two registration
        messages.warning(request, 'You need to complete the registration process first')
        return redirect('home')

@login_required
def level_two_complete(request):
    """Mark level two as complete and redirect to level three."""
    auth_session = AuthenticationSession.objects.filter(
        user=request.user,
        session_key=request.session.session_key
    ).first()
    
    if auth_session:
        auth_session.level_two_complete = True
        auth_session.save()
        logger.info(f"User {request.user.username} successfully authenticated level two")
        
        return redirect('level_three')
    
    messages.error(request, 'Authentication session not found')
    return redirect('level_one')

@login_required
def level_three(request):
    """Level three authentication using gesture recognition."""
    # Check if levels one and two are complete
    auth_session = AuthenticationSession.objects.filter(
        user=request.user,
        session_key=request.session.session_key
    ).first()
    
    if not auth_session or not auth_session.level_one_complete or not auth_session.level_two_complete:
        messages.error(request, 'You must complete previous authentication levels first')
        return redirect('level_one')
    
    # Check if user has a gesture registered
    profile, created = BiometricProfile.objects.get_or_create(user=request.user)
    
    if not profile.gesture_registered:
        messages.warning(request, 'You need to register your gesture first')
        return redirect('register_gesture_direct', user_id=request.user.username)
    
    # Directly redirect to gesture verification
    return redirect('verify_gesture_direct', user_id=request.user.username)

@login_required
def level_three_complete(request):
    """Mark level three as complete and redirect to dashboard."""
    auth_session = AuthenticationSession.objects.filter(
        user=request.user,
        session_key=request.session.session_key
    ).first()
    
    if auth_session:
        auth_session.level_three_complete = True
        auth_session.save()
        logger.info(f"User {request.user.username} successfully authenticated level three")
        
        return redirect('dashboard')
    
    messages.error(request, 'Authentication session not found')
    return redirect('level_one')

@login_required
def dashboard(request):
    """Dashboard view after successful multi-level authentication."""
    auth_session = AuthenticationSession.objects.filter(
        user=request.user,
        session_key=request.session.session_key
    ).first()
    
    if not auth_session or not auth_session.is_fully_authenticated:
        messages.error(request, 'You must complete all authentication levels first')
        return redirect('level_one')
    
    return render(request, 'face_gesture/dashboard.html')

@login_required
def auth_status(request):
    """View to display authentication status and progress."""
    auth_session = AuthenticationSession.objects.filter(
        user=request.user,
        session_key=request.session.session_key
    ).first()
    
    if not auth_session:
        # Create a new session if one doesn't exist
        auth_session = AuthenticationSession.objects.create(
            user=request.user,
            session_key=request.session.session_key,
            level_one_complete=False,
            level_two_complete=False,
            level_three_complete=False
        )
    
    # Get the biometric profile for this user
    profile, created = BiometricProfile.objects.get_or_create(user=request.user)
    
    context = {
        'auth_session': auth_session,
        'biometric_profile': profile,
    }
    
    return render(request, 'face_gesture/auth_status.html', context)

def register_face(request, user_id):
    """View for registering a new face."""
    if request.method == 'POST':
        if 'frame' not in request.FILES:
            return JsonResponse({
                'status': 'error',
                'message': 'No frame data received'
            }, status=400)

        try:
            # Process the frame data
            frame_data = request.FILES['frame'].read()
            frame = process_frame_data(frame_data)
            
            if frame is None:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Could not process frame data'
                }, status=400)
            
            # Get capture number
            capture_number = int(request.POST.get('capture_number', 1))
            
            # Initialize face verifier
            verifier = FaceVerifier()
            
            # Ensure the user directory exists
            user_dir = os.path.join('faces', user_id)
            os.makedirs(user_dir, exist_ok=True)
            
            # Save individual capture
            capture_path = os.path.join(user_dir, f"{user_id}_capture_{capture_number}.jpg")
            cv2.imwrite(capture_path, frame)
            
            # Process the frame for face encoding
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            
            if not face_locations:
                return JsonResponse({
                    'status': 'error',
                    'message': 'No face detected in frame'
                }, status=400)
            
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            if not face_encodings:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Could not generate face encoding'
                }, status=400)
            
            # Save individual encoding
            encoding_path = os.path.join(user_dir, f"{user_id}_encoding_{capture_number}.npy")
            np.save(encoding_path, face_encodings[0])
            
            # If this is the final capture, create average encoding
            if capture_number == 5:
                try:
                    # Load all encodings
                    encodings = []
                    for i in range(1, 6):
                        enc_path = os.path.join(user_dir, f"{user_id}_encoding_{i}.npy")
                        encodings.append(np.load(enc_path))
                    
                    # Calculate average encoding
                    average_encoding = np.mean(encodings, axis=0)
                    
                    # Save final encoding
                    final_encoding_path = os.path.join(user_dir, f"{user_id}_encoding.npy")
                    np.save(final_encoding_path, average_encoding)
                    
                    # Clean up individual encodings
                    for i in range(1, 6):
                        enc_path = os.path.join(user_dir, f"{user_id}_encoding_{i}.npy")
                        if os.path.exists(enc_path):
                            os.remove(enc_path)
                except Exception as e:
                    logger.error(f"Error creating average encoding: {str(e)}")
                    return JsonResponse({
                        'status': 'error',
                        'message': 'Error creating final face encoding'
                    }, status=500)
            
            return JsonResponse({
                'status': 'success',
                'message': f'Capture {capture_number} successful'
            })
                
        except Exception as e:
            logger.error(f"Error during registration: {str(e)}")
            return JsonResponse({
                'status': 'error',
                'message': f'Error during registration: {str(e)}'
            }, status=500)
    
    return render(request, 'face_gesture/register.html', {'user_id': user_id})

def verify_face(request, user_id):
    """View for verifying a face."""
    if request.method == 'POST':
        if 'frame' not in request.FILES:
            return JsonResponse({
                'status': 'error',
                'message': 'No frame data received'
            }, status=400)

        try:
            # Process the frame data
            frame_data = request.FILES['frame'].read()
            frame = process_frame_data(frame_data)
            
            if frame is None:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Could not process frame data'
                }, status=400)
            
            # Initialize face verifier
            verifier = FaceVerifier()
            
            # Load the known face encoding
            if not verifier.load_known_face(user_id):
                return JsonResponse({
                    'status': 'error',
                    'message': 'No registered face found for this user. Please register first.'
                }, status=400)
            
            # Process the frame for verification
            frame, result, distance, face_location = verifier.process_verification_frame(frame)
            
            # Convert face location to JSON-serializable format
            face_location_dict = None
            if face_location:
                top, right, bottom, left = face_location
                face_location_dict = {
                    'top': int(top),
                    'right': int(right),
                    'bottom': int(bottom),
                    'left': int(left)
                }
            
            # Convert numpy types to Python native types
            response_data = {
                'status': 'success',
                'message': 'Face processed',
                'matched': bool(result) if result is not None else False,
                'distance': float(distance) if distance is not None else None,
                'face_location': face_location_dict
            }
            
            if not face_location_dict:
                response_data['message'] = 'No face detected in frame'
            elif response_data['matched']:
                response_data['message'] = 'Face matched successfully'
                
                # Update BiometricProfile for the authenticated user if logged in
                if request.user.is_authenticated and request.user.username == user_id:
                    profile, created = BiometricProfile.objects.get_or_create(user=request.user)
                    profile.face_registered = True
                    profile.save()
                    
                    # Add redirect URL for level completion if this is part of the auth flow
                    auth_session = AuthenticationSession.objects.filter(
                        user=request.user,
                        session_key=request.session.session_key,
                        level_one_complete=True
                    ).first()
                    
                    if auth_session and not auth_session.level_two_complete:
                        # Complete level two and redirect to level three (gesture verification)
                        auth_session.level_two_complete = True
                        auth_session.save()
                        logger.info(f"User {request.user.username} successfully authenticated level two")
                        response_data['redirect_url'] = '/level-three/'
                    elif auth_session and auth_session.level_two_complete:
                        # If already completed level two and this is face verification, redirect to gesture verification
                        # Check if gesture is registered for the user
                        if profile.gesture_registered:
                            response_data['redirect_url'] = reverse('verify_gesture_direct', kwargs={'user_id': user_id})
                        else:
                            # If no gesture registered yet, redirect to gesture registration
                            response_data['redirect_url'] = reverse('register_gesture_direct', kwargs={'user_id': user_id})
            else:
                response_data['message'] = 'Face did not match'
            
            return JsonResponse(response_data)
                
        except Exception as e:
            logger.error(f"Error during verification: {str(e)}")
            return JsonResponse({
                'status': 'error',
                'message': f'Error during verification: {str(e)}'
            }, status=500)
    
    return render(request, 'face_gesture/verify.html', {'user_id': user_id})

# def gesture_registration_page(request, user_id):
#     """View for rendering the gesture registration page."""
#     return render(request, 'face_gesture/gesture_register.html', {'user_id': user_id})

# def gesture_verification_page(request, user_id):
#     """View for rendering the gesture verification page."""
#     return render(request, 'face_gesture/gesture_verify.html', {'user_id': user_id})

def register_gesture(request):
    """API endpoint for registering a gesture."""
    if not request.method == 'POST':
        return JsonResponse({
            'status': 'error',
            'message': 'Only POST method is allowed'
        }, status=405)

    user_id = request.POST.get('user_id')
    if not user_id:
        return JsonResponse({
            'status': 'error',
            'message': 'user_id is required'
        }, status=400)

    if 'frame' not in request.FILES:
        return JsonResponse({
            'status': 'error',
            'message': 'No frame data received'
        }, status=400)

    try:
        # Process the frame data
        frame_data = request.FILES['frame'].read()
        frame = process_frame_data(frame_data)
        
        if frame is None:
            return JsonResponse({
                'status': 'error',
                'message': 'Could not process frame data'
            }, status=400)
        
        # Get capture number
        capture_number = int(request.POST.get('capture_number', 1))
        logger.info(f"Processing capture {capture_number} for user {user_id}")
        
        # Ensure the user directory exists
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        user_dir = os.path.join(base_dir, 'gestures', user_id)
        os.makedirs(user_dir, exist_ok=True)
        logger.info(f"Using directory: {user_dir}")
        
        # Process the frame for hand landmarks
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        with mp_hands.Hands(
            min_tracking_confidence=0.7,
            max_num_hands=2,
            min_detection_confidence=0.7
        ) as hands:
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            # Convert back to BGR for OpenCV operations
            frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            if not results.multi_hand_landmarks:
                return JsonResponse({
                    'status': 'error',
                    'message': 'No hand detected in frame'
                }, status=400)
            
            # Draw hand landmarks on frame
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
            
            # Normalize and save landmarks
            landmarks = results.multi_hand_landmarks[0].landmark
            normalized_vector = GestureRegistrar().normalize_landmarks(landmarks)

            
            # Save individual landmarks with debug info
            landmarks_path = os.path.join(user_dir, f"{user_id}_landmarks_{capture_number}.npy")
            np.save(landmarks_path, normalized_vector)
            logger.info(f"Saved landmarks to: {landmarks_path}")
            
            # Save the frame with landmarks drawn
            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = buffer.tobytes()
            
            # If this is the final capture, create average landmarks
            if capture_number == 5:
                try:
                    logger.info("Processing final capture and creating average landmarks")
                    # Load all landmarks
                    all_landmarks = []
                    for i in range(1, 6):
                        lm_path = os.path.join(user_dir, f"{user_id}_landmarks_{i}.npy")
                        if not os.path.exists(lm_path):
                            raise FileNotFoundError(f"Missing landmark file: {lm_path}")
                        all_landmarks.append(np.load(lm_path))
                        logger.info(f"Loaded landmark file {i}: {lm_path}")
                    
                    # Calculate average landmarks
                    average_landmarks = np.mean(all_landmarks, axis=0)
                    
                    # Save final landmarks
                    final_landmarks_path = os.path.join(user_dir, f"{user_id}_gesture.npy")
                    np.save(final_landmarks_path, average_landmarks)
                    logger.info(f"Saved final gesture to: {final_landmarks_path}")
                    
                    # Clean up individual landmarks
                    for i in range(1, 6):
                        lm_path = os.path.join(user_dir, f"{user_id}_landmarks_{i}.npy")
                        if os.path.exists(lm_path):
                            os.remove(lm_path)
                            logger.info(f"Cleaned up temporary file: {lm_path}")
                    
                    return JsonResponse({
                        'status': 'success',
                        'message': 'Gesture registration completed successfully ',
                        #'frame': frame_data.decode('latin1')
                    })
                    
                except Exception as e:
                    logger.error(f"Error creating average landmarks: {str(e)}")
                    return JsonResponse({
                        'status': 'error',
                        'message': f'Error creating final gesture landmarks: {str(e)}'
                    }, status=500)
            
            return JsonResponse({
                'status': 'success',
                'message': f'Capture {capture_number} successful',
                'frame': frame_data.decode('latin1')
            })
                
    except Exception as e:
        logger.error(f"Error during gesture registration: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'message': f'Error during registration: {str(e)}'
        }, status=500)

def normalize_landmarks(landmarks):
    """Normalize hand landmarks to be invariant to translation and scale."""
    # Convert landmarks to numpy array
    points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    
    # Center the points by subtracting the mean
    centered = points - np.mean(points, axis=0)
    
    # Scale to unit size
    scale = np.sqrt(np.sum(centered**2))
    normalized = centered / scale if scale != 0 else centered
    
    # Flatten to 1D vector
    return normalized.flatten()

def verify_gesture(request):
    """API endpoint for verifying a gesture."""
    if not request.method == 'POST':
        return JsonResponse({
            'status': 'error',
            'message': 'Only POST method is allowed'
        }, status=405)

    user_id = request.POST.get('user_id')
    if not user_id:
        return JsonResponse({
            'status': 'error',
            'message': 'user_id is required'
        }, status=400)

    if 'frame' not in request.FILES:
        return JsonResponse({
            'status': 'error',
            'message': 'No frame data received'
        }, status=400)

    try:
        # Process the frame data
        frame_data = request.FILES['frame'].read()
        frame = process_frame_data(frame_data)
        
        if frame is None:
            return JsonResponse({
                'status': 'error',
                'message': 'Could not process frame data'
            }, status=400)
        
        # Load the known gesture landmarks
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        gesture_path = os.path.join(base_dir, 'gestures', user_id, f"{user_id}_gesture.npy")
        logger.info(f"Looking for gesture file at: {gesture_path}")
        
        try:
            known_gesture = np.load(gesture_path)
            logger.info("Successfully loaded gesture file")
        except FileNotFoundError:
            logger.error(f"No gesture file found at: {gesture_path}")
            return JsonResponse({
                'status': 'error',
                'message': 'No registered gesture found for this user. Please register first.'
            }, status=400)
        
        # Process the frame for hand landmarks
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.7
        ) as hands:
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            # Convert back to BGR for OpenCV operations
            frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            if not results.multi_hand_landmarks:
                return JsonResponse({
                    'status': 'error',
                    'message': 'No hand detected in frame'
                }, status=400)
            
            # Draw hand landmarks on frame
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
            
            # Normalize current landmarks
            landmarks = results.multi_hand_landmarks[0].landmark
            current_vector = normalize_landmarks(landmarks)
            
            # Calculate similarity
            similarity = np.dot(current_vector, known_gesture) / (
                np.linalg.norm(current_vector) * np.linalg.norm(known_gesture)
            )
            logger.info(f"Calculated similarity: {similarity:.2%}")
            
            # Save the frame with landmarks drawn
            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = buffer.tobytes()
            
            # Convert numpy types to Python native types
            response_data = {
                'status': 'success',
                'message': 'Gesture processed',
                'matched': bool(similarity > 0.85),
                'similarity': float(similarity),
                'frame': frame_data.decode('latin1')
            }
            
            if response_data['matched']:
                logger.info("Gesture matched successfully")
                response_data['message'] = 'Gesture matched successfully'
            else:
                logger.info("Gesture did not match")
                response_data['message'] = 'Gesture did not match'
            
            return JsonResponse(response_data)
                
    except Exception as e:
        logger.error(f"Error during gesture verification: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'message': f'Error during verification: {str(e)}'
        }, status=500)

def register_gesture_direct(request, user_id):
    """View for direct gesture registration using MediaPipe."""
    if request.method == 'GET':
        return render(request, 'face_gesture/gesture_register_direct.html', {'user_id': user_id})
    else:
        # For AJAX or direct API calls
        try:
            registrar = GestureRegistrar()
            success, save_file, consistency = registrar.register_gesture(user_id)
            
            if success:
                return JsonResponse({
                    'status': 'success',
                    'message': f'Gesture registered successfully! Consistency: {consistency:.2%}',
                    'saved_to': save_file
                })
            else:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Gesture registration failed. Please try again.'
                }, status=400)
                
        except Exception as e:
            logger.error(f"Error during direct gesture registration: {str(e)}")
            return JsonResponse({
                'status': 'error',
                'message': f'Error during registration: {str(e)}'
            }, status=500)

def verify_gesture_direct(request, user_id):
    """View for direct gesture verification using MediaPipe."""
    if request.method == 'GET':
        return render(request, 'face_gesture/gesture_verify_direct.html', {'user_id': user_id})
    else:
        # For AJAX or direct API calls
        try:
            verifier = GestureVerifier()
            success, similarity = verifier.verify_gesture(user_id)
            
            response_data = {}
            
            if success:
                response_data = {
                    'status': 'success',
                    'message': f'Gesture verified successfully! Similarity: {similarity:.2%}'
                }
                
                # Update BiometricProfile for the authenticated user if logged in
                if request.user.is_authenticated and request.user.username == user_id:
                    profile, created = BiometricProfile.objects.get_or_create(user=request.user)
                    profile.gesture_registered = True
                    profile.save()
                    
                    # Add redirect URL for level completion if this is part of the auth flow
                    auth_session = AuthenticationSession.objects.filter(
                        user=request.user,
                        session_key=request.session.session_key,
                        level_one_complete=True,
                        level_two_complete=True
                    ).first()
                    
                    if auth_session and not auth_session.level_three_complete:
                        # Complete level three here directly
                        auth_session.level_three_complete = True
                        auth_session.save()
                        logger.info(f"User {request.user.username} successfully authenticated level three")
                        
                        # This is the final step of authentication, redirect to dashboard
                        response_data['redirect_url'] = reverse('dashboard')
                        
                        # Add a success message that will be displayed on the dashboard
                        messages.success(request, 'Multi-level authentication completed successfully!')
            else:
                # If similarity is a string, it's an error message
                if isinstance(similarity, str):
                    message = similarity
                else:
                    message = f'Gesture verification failed. Similarity: {similarity:.2%}'
                
                response_data = {
                    'status': 'error',
                    'message': message
                }
                
            return JsonResponse(response_data)
                
        except Exception as e:
            logger.error(f"Error during direct gesture verification: {str(e)}")
            return JsonResponse({
                'status': 'error',
                'message': f'Error during verification: {str(e)}'
            }, status=500)

@login_required
def pattern_authentication(request):
    """Level two authentication using pattern matching."""
    # Check if level one is complete
    auth_session = AuthenticationSession.objects.filter(
        user=request.user,
        session_key=request.session.session_key
    ).first()
    
    if not auth_session or not auth_session.level_one_complete:
        messages.error(request, 'You must complete level one authentication first')
        return redirect('level_one')
    
    # Check if user has a pattern registered
    try:
        user_profile = UserProfile.objects.get(user=request.user)
        if not user_profile.pattern:
            messages.warning(request, 'You need to complete registration first')
            return redirect('home')
            
        # Generate 9 grid items (1-9)
        grid_items = list(range(1, 10))
        
        # Add a message explaining what to do
        messages.info(request, 'Please recreate your exact pattern from registration. Place each piece in the same position you placed it during registration.')
        
        return render(request, 'face_gesture/login_pattern_level_two.html', {
            'grid_items': grid_items,
            'pattern_image_url': reverse('get_pattern_image')
        })
    except UserProfile.DoesNotExist:
        messages.error(request, 'Pattern not found. Please complete registration first.')
        return redirect('home')

def get_pattern_image(request):
    """Serve the user's pattern image securely."""
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Authentication required'}, status=401)
    
    try:
        user_profile = UserProfile.objects.get(user=request.user)
        
        # Return the image data
        if user_profile.pattern_image:
            return HttpResponse(
                user_profile.pattern_image, 
                content_type='image/jpeg'
            )
        else:
            return JsonResponse({'error': 'Pattern image not found'}, status=404)
    except UserProfile.DoesNotExist:
        return JsonResponse({'error': 'Profile not found'}, status=404)
    except Exception as e:
        logger.error(f"Error retrieving pattern image: {str(e)}")
        return JsonResponse({'error': 'Error retrieving pattern image'}, status=500)

@require_POST
def verify_pattern(request):
    """API endpoint for verifying pattern authentication."""
    if not request.user.is_authenticated:
        return JsonResponse({
            'status': 'error',
            'message': 'Authentication required'
        }, status=401)
    
    try:
        # Get the submitted pattern from request body
        data = json.loads(request.body)
        submitted_pattern = data.get('pattern', {})
        
        if not submitted_pattern or 'sequence' not in submitted_pattern:
            return JsonResponse({
                'status': 'error',
                'message': 'No valid pattern submitted'
            }, status=400)
        
        # Get the user's stored pattern
        user_profile = UserProfile.objects.get(user=request.user)
        stored_pattern_data = json.loads(user_profile.pattern)
        
        # Get the stored sequence - this is the sequence of blockIds in order of placement during registration
        stored_sequence = stored_pattern_data.get('sequence', [])
        
        # Get the submitted sequence - this is the sequence of blockIds in order of placement during login
        submitted_sequence = submitted_pattern.get('sequence', [])
        
        if not stored_sequence or not submitted_sequence:
            return JsonResponse({
                'status': 'error',
                'message': 'Invalid pattern data'
            }, status=400)
        
        # Log the sequences for debugging
        logger.info(f"Verifying pattern for user {request.user.username}")
        logger.info(f"Stored sequence: {stored_sequence}")
        logger.info(f"Submitted sequence: {submitted_sequence}")
        
        # Convert stored sequence to a list of blockIds in placement order
        stored_sequence_ids = []
        for item in stored_sequence:
            if isinstance(item, dict) and 'blockId' in item:
                stored_sequence_ids.append(str(item['blockId']))
        
        # Convert submitted sequence to a list of blockIds (it might already be a list of strings)
        submitted_sequence_ids = [str(item) for item in submitted_sequence]
            
        logger.info(f"Stored sequence order: {stored_sequence_ids}")
        logger.info(f"Submitted sequence order: {submitted_sequence_ids}")
        
        # Check if the sequence orders match exactly
        if stored_sequence_ids != submitted_sequence_ids:
            logger.warning(f"Pattern verification failed. Sequence order doesn't match.")
            logger.warning(f"Expected sequence: {stored_sequence_ids}")
            logger.warning(f"Submitted sequence: {submitted_sequence_ids}")
            return JsonResponse({
                'status': 'error',
                'message': 'Pattern verification failed. Please place pieces in the exact same sequence as during registration.'
            }, status=400)
        
        # If we want to also check final positions (additional security check)
        # Create a mapping of which piece is in which position
        stored_pattern_pos = {}
        for item in stored_sequence:
            if isinstance(item, dict):
                blockId = item.get('blockId')
                position = item.get('position')
                if blockId is not None and position is not None:
                    stored_pattern_pos[str(blockId)] = str(position)
        
        # Check final positions - use placementObjects to get positions based on blockId
        if 'placements' in submitted_pattern:
            submitted_pattern_pos = {}
            for placement in submitted_pattern['placements']:
                blockId = placement.get('gridPosition')
                position = placement.get('targetPosition')
                if blockId is not None and position is not None:
                    submitted_pattern_pos[str(blockId)] = str(position)
                
            # Log the position maps for debugging
            logger.info(f"Stored positions: {stored_pattern_pos}")
            logger.info(f"Submitted positions: {submitted_pattern_pos}")
            
            # Check if the final positions match
            if stored_pattern_pos != submitted_pattern_pos:
                logger.warning(f"Pattern verification failed. Positions don't match.")
                logger.warning(f"Expected positions: {stored_pattern_pos}")
                logger.warning(f"Submitted positions: {submitted_pattern_pos}")
                return JsonResponse({
                    'status': 'error',
                    'message': 'Pattern verification failed. Please place pieces in the exact same positions as during registration.'
                }, status=400)
        
        # If we got here, the sequence and (optionally) positions match - authentication succeeds
        # Update authentication session
        auth_session = AuthenticationSession.objects.filter(
            user=request.user,
            session_key=request.session.session_key
        ).first()
        
        if auth_session:
            auth_session.level_two_complete = True
            auth_session.save()
            logger.info(f"User {request.user.username} successfully authenticated level two with pattern")
        
        # Redirect to facial verification
        return JsonResponse({
            'status': 'success',
            'message': 'Pattern verification successful',
            'redirect_url': reverse('verify_face', kwargs={'user_id': request.user.username})
        })
        
    except UserProfile.DoesNotExist:
        return JsonResponse({
            'status': 'error',
            'message': 'User profile not found'
        }, status=404)
    except json.JSONDecodeError:
        return JsonResponse({
            'status': 'error',
            'message': 'Invalid pattern data'
        }, status=400)
    except Exception as e:
        logger.error(f"Error during pattern verification: {str(e)}")
        logger.error(traceback.format_exc())
        return JsonResponse({
            'status': 'error',
            'message': f'Error during verification: {str(e)}'
        }, status=500)