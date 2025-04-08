from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse, StreamingHttpResponse
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from .models import UserProfile, AuthenticationSession, BiometricData
from .forms import UserRegistrationForm
from django.core.files.storage import FileSystemStorage
import random
import logging
import json
from datetime import datetime
import cv2
import numpy as np
import face_recognition
import mediapipe as mp
import time
from django.urls import reverse
import threading
import queue
from PIL import Image
from io import BytesIO
import base64

logger = logging.getLogger(__name__)

def index(request):
    """
    Main landing page view.
    If user is authenticated, redirect to dashboard.
    Otherwise, show the index page.
    """
    if request.user.is_authenticated:
        return redirect('dashboard')
    return render(request, 'index.html')

def register(request):
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
    
    return render(request, 'register.html', {'form': form})

def register_level_two(request):
    if request.method == 'POST':
        pattern = json.loads(request.POST.get('pattern'))
        sorted_pattern = sorted(pattern, key=lambda x: int(x.get('placementTime', 0)))
        
        # Get the uploaded image
        if 'image' not in request.FILES:
            return JsonResponse({'status': 'error', 'message': 'No image uploaded'})
            
        image = request.FILES['image']
        
        try:
            # Get registration data from session
            registration_data = request.session.get('registration_data')
            if not registration_data:
                logger.error("No registration data found in session")
                return JsonResponse({'status': 'error', 'message': 'Registration data not found in session'})
            
            # Create user
            form = UserRegistrationForm(registration_data)
            if form.is_valid():
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
                
                # Read and process image
                # Open image using PIL
                img = Image.open(image)
                
                # Resize image to a reasonable size (e.g., 800x800)
                max_size = (800, 800)
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save to BytesIO
                img_io = BytesIO()
                img.save(img_io, format='JPEG', quality=85)
                img_io.seek(0)
                
                # Create and save user profile with processed image data
                user_profile = UserProfile.objects.create(
                    user=user,
                    pattern_image=img_io.getvalue(),  # Store processed binary data
                    pattern=json.dumps(sorted_pattern)
                )
                logger.info("User profile created successfully")
                
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
                        raise Exception("Failed to create or find authentication session")
                
                # Clear the session data
                if 'registration_data' in request.session:
                    del request.session['registration_data']
                
                # Get the redirect URL and log it
                redirect_url = reverse('register_level_three')
                logger.info(f"Redirecting to: {redirect_url}")
                
                return JsonResponse({
                    'status': 'success',
                    'message': 'Pattern successfully registered!',
                    'redirect_url': redirect_url
                })
            else:
                logger.error(f"Form validation errors: {form.errors}")
                return JsonResponse({
                    'status': 'error',
                    'message': f'Invalid registration data: {form.errors}'
                })
            
        except Exception as e:
            logger.error(f"Error in register_level_two: {str(e)}")
            return JsonResponse({'status': 'error', 'message': str(e)})
    
    return render(request, 'register_level_two.html')



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
    
    # Randomly select and shuffle 9 images
    selected_images = random.sample(image_options, 9)
    random.shuffle(selected_images)
    
    return selected_images

def level_one(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(username=username, password=password)
        
        if user is not None:
            # Create session for tracking authentication progress
            request.session['level_one_complete'] = True
            request.session['user_id'] = user.id
            return redirect('level_two')
        else:
            messages.error(request, 'Invalid credentials')
    
    return render(request, 'login.html')  # Changed from level_one.html to login.html

@login_required
def level_two(request):
    session = AuthenticationSession.objects.filter(
        user=request.user,
        level_one_complete=True,
        session_key=request.session.session_key
    ).first()
    
    if not session:
        return redirect('level_one')

    user_profile = UserProfile.objects.get(user=request.user)

    if request.method == 'POST':
        submitted_pattern = request.POST.get('pattern')
        
        if submitted_pattern == user_profile.pattern:
            session.level_two_complete = True
            session.save()
            return redirect('level_three')
        else:
            messages.error(request, 'Incorrect pattern')

    context = {
        'auth_image_url': user_profile.auth_image.url,
    }
    return render(request, 'level_two.html', context)

# Add this new view for progress updates
def registration_progress(request):
    def event_stream():
        while True:
            if not hasattr(request.session, 'registration_updates'):
                request.session['registration_updates'] = queue.Queue()
            
            try:
                update = request.session['registration_updates'].get(timeout=1)
                yield f"data: {json.dumps(update)}\n\n"
            except queue.Empty:
                yield "data: {}\n\n"
    
    response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    return response

def render_level_three_page(request):
    return render(request, 'register_level_three.html')

class GestureRecognition:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.2,  # Lower threshold for easier detection
            min_tracking_confidence=0.2
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.gesture_sequence = []
        self.required_gesture_count = 3
        self.last_gesture_time = time.time()
        self.gesture_timeout = 0.5  # Reduced timeout for more frequent detection

    def detect_gesture(self, hand_landmarks):
        try:
            # Get key points
            thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            thumb_ip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
            middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
            ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

            # Log landmark positions for debugging
            logger.info(f"Thumb tip position: y={thumb_tip.y:.3f}")
            logger.info(f"Index tip position: y={index_tip.y:.3f}")
            logger.info(f"Middle tip position: y={middle_tip.y:.3f}")

            # More lenient thresholds
            vertical_threshold = 0.1
            horizontal_threshold = 0.15

            # Thumbs up detection - more lenient
            if (thumb_tip.y < thumb_ip.y - vertical_threshold and
                thumb_tip.y < index_tip.y and
                thumb_tip.y < middle_tip.y):
                logger.info("Detected: thumbs_up")
                return "thumbs_up"
            
            # Thumbs down detection - more lenient
            elif (thumb_tip.y > thumb_ip.y + vertical_threshold and
                  thumb_tip.y > index_tip.y and
                  thumb_tip.y > middle_tip.y):
                logger.info("Detected: thumbs_down")
                return "thumbs_down"
            
            # Victory sign detection - more lenient
            elif (index_tip.y < index_pip.y - vertical_threshold and  # Index finger up
                  middle_tip.y < middle_pip.y - vertical_threshold and  # Middle finger up
                  abs(index_tip.y - middle_tip.y) < horizontal_threshold and  # Roughly same height
                  ring_tip.y > middle_pip.y and  # Ring finger down
                  pinky_tip.y > middle_pip.y):  # Pinky down
                logger.info("Detected: victory")
                return "victory"

            # Log why no gesture was detected
            logger.info("No gesture detected:")
            logger.info(f"Thumb above index: {thumb_tip.y < index_tip.y}")
            logger.info(f"Index up: {index_tip.y < index_pip.y}")
            logger.info(f"Middle up: {middle_tip.y < middle_pip.y}")
            return None

        except Exception as e:
            logger.error(f"Error in detect_gesture: {str(e)}")
            return None


def register_level_three(request):
    # Check if user has completed previous registration steps
    if not request.user.is_authenticated:
        messages.error(request, 'Please log in first.')
        return redirect('level_one')

    # Check if level two is complete by checking AuthenticationSession
    auth_session = AuthenticationSession.objects.filter(
        user=request.user,
        level_one_complete=True,
        level_two_complete=True,
        session_key=request.session.session_key
    ).first()

    if not auth_session:
        messages.error(request, 'Please complete previous registration steps first.')
        return redirect('register')

    # Check if biometric data already exists
    existing_biometric = BiometricData.objects.filter(user=request.user).first()
    if existing_biometric:
        logger.info(f"Existing biometric data found for user {request.user.username}")
        logger.info(f"Face encoding exists: {len(existing_biometric.face_encoding) > 0}")
        logger.info(f"Gesture sequence: {existing_biometric.gesture_sequence}")

    if request.method == 'POST':
        try:
            logger.info(f"Starting biometric registration for user: {request.user.username}")
            
            # Get the frame data from the POST request
            frame_data = request.FILES.get('frame')
            if not frame_data:
                return JsonResponse({
                    'status': 'error',
                    'message': 'No frame data received'
                })

            # Convert the frame data to an image
            frame_image = Image.open(frame_data)
            frame_array = np.array(frame_image)
            rgb_frame = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)

            # Initialize response data
            response_data = {
                'status': 'success',
                'face_detected': False,
                'debug_info': {}  # Add debug information
            }

            # Face Detection with HOG model
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            logger.info(f"Number of faces detected: {len(face_locations)}")
            
            # Get or create biometric data
            biometric_data, created = BiometricData.objects.get_or_create(
                user=request.user,
                defaults={
                    'face_encoding': b'',
                    'gesture_sequence': [],
                    'is_active': True
                }
            )

            # Process face detection
            if len(face_locations) > 0:
                response_data['face_detected'] = True
                response_data['face_location'] = face_locations[0]

                # Only process face encoding if we don't have one yet
                if not biometric_data.face_encoding:
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    if face_encodings:
                        face_encoding = face_encodings[0]
                        biometric_data.face_encoding = face_encoding.tobytes()
                        biometric_data.save()
                        logger.info(f"Face encoding saved for user: {request.user.username}")
                        logger.info(f"Face encoding length: {len(biometric_data.face_encoding)}")
                        response_data['debug_info']['face_encoding_saved'] = True
                        response_data['debug_info']['face_encoding_length'] = len(biometric_data.face_encoding)

            # Initialize gesture recognition with more lenient parameters
            gesture_recognition = GestureRecognition()
            hand_results = gesture_recognition.hands.process(rgb_frame)
            
            if hand_results.multi_hand_landmarks:
                logger.info(f"Detected {len(hand_results.multi_hand_landmarks)} hands")
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    # Add hand landmarks to response for visualization
                    response_data['hand_landmarks'] = {
                        'points': [
                            {'x': landmark.x, 'y': landmark.y}
                            for landmark in hand_landmarks.landmark
                        ],
                        'connections': gesture_recognition.mp_hands.HAND_CONNECTIONS
                    }
                    
                    current_gesture = gesture_recognition.detect_gesture(hand_landmarks)
                    if current_gesture:
                        logger.info(f"Detected gesture: {current_gesture}")
                        if not biometric_data.gesture_sequence:
                            biometric_data.gesture_sequence = []
                            
                        if current_gesture not in biometric_data.gesture_sequence:
                            biometric_data.gesture_sequence.append(current_gesture)
                            biometric_data.save()
                            logger.info(f"New gesture recorded: {current_gesture}")
                            logger.info(f"Current gesture sequence: {biometric_data.gesture_sequence}")
                            
                            response_data.update({
                                'gesture_detected': True,
                                'gesture': current_gesture,
                                'gesture_count': len(biometric_data.gesture_sequence),
                                'debug_info': {
                                    'current_gesture': current_gesture,
                                    'gesture_sequence': biometric_data.gesture_sequence
                                }
                            })
                            break
            else:
                logger.info("No hands detected in the frame")
                response_data['debug_info']['hand_detection'] = 'No hands detected'

            # Check if registration is complete
            if len(biometric_data.gesture_sequence) >= 3:
                response_data['registration_complete'] = True
                response_data['redirect_url'] = reverse('dashboard')
                logger.info(f"Registration completed for user: {request.user.username}")
                logger.info(f"Final gesture sequence: {biometric_data.gesture_sequence}")

            # Add current registration status to response
            response_data['debug_info']['current_status'] = {
                'face_saved': len(biometric_data.face_encoding) > 0,
                'gestures_saved': len(biometric_data.gesture_sequence),
                'gestures_needed': 3 - len(biometric_data.gesture_sequence)
            }

            return JsonResponse(response_data)

        except Exception as e:
            logger.error(f"Error in register_level_three: {str(e)}")
            return JsonResponse({
                'status': 'error',
                'message': f'An error occurred: {str(e)}'
            })

    return render(request, 'register_level_three.html')


@login_required
def verify_level_three(request):
    if request.method == 'POST':
        try:
            # Get user's stored biometric data
            stored_biometric = BiometricData.objects.get(user=request.user)
            stored_face_encoding = np.frombuffer(stored_biometric.face_encoding)
            stored_gesture_sequence = stored_biometric.gesture_sequence

            # Initialize video capture and gesture recognition
            cap = cv2.VideoCapture(0)
            gesture_recognition = GestureRecognition()
            
            face_verified = False
            gestures_verified = False
            
            while not (face_verified and gestures_verified):
                ret, frame = cap.read()
                if not ret:
                    break
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Face verification
                face_locations = face_recognition.face_locations(rgb_frame)
                if len(face_locations) == 1:
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                    # Compare with stored face encoding
                    matches = face_recognition.compare_faces([stored_face_encoding], face_encodings, tolerance=0.6)
                    if matches[0]:
                        face_verified = True
                        cv2.putText(frame, "Face Verified!", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Gesture verification
                hand_results = gesture_recognition.hands.process(rgb_frame)
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        gesture_recognition.mp_draw.draw_landmarks(
                            frame, hand_landmarks, gesture_recognition.mp_hands.HAND_CONNECTIONS)
                        
                        current_gesture = gesture_recognition.detect_gesture(hand_landmarks)
                        current_time = time.time()
                        
                        if current_gesture and (current_time - gesture_recognition.last_gesture_time) > gesture_recognition.gesture_timeout:
                            gesture_recognition.gesture_sequence.append(current_gesture)
                            gesture_recognition.last_gesture_time = current_time
                            
                            # Check if current gesture sequence matches stored sequence
                            if len(gesture_recognition.gesture_sequence) == len(stored_gesture_sequence):
                                if gesture_recognition.gesture_sequence == stored_gesture_sequence:
                                    gestures_verified = True
                                else:
                                    gesture_recognition.gesture_sequence = []  # Reset if wrong
                
                # Display verification status
                cv2.putText(frame, f"Face: {'Verified' if face_verified else 'Not Verified'}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if face_verified else (0, 0, 255), 2)
                cv2.putText(frame, f"Gestures: {len(gesture_recognition.gesture_sequence)}/{len(stored_gesture_sequence)}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Verification - Face and Gesture', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
            if face_verified and gestures_verified:
                # Update authentication session
                session = AuthenticationSession.objects.get(
                    user=request.user,
                    session_key=request.session.session_key
                )
                session.level_three_complete = True
                session.save()
                
                messages.success(request, 'Biometric verification successful!')
                return redirect('dashboard')
            else:
                messages.error(request, 'Verification failed. Please try again.')
                
        except BiometricData.DoesNotExist:
            messages.error(request, 'No biometric data found. Please register first.')
        except Exception as e:
            logger.error(f"Error in verify_level_three: {str(e)}")
            messages.error(request, f'Verification failed: {str(e)}')
    
    return render(request, 'verify_level_three.html')

@login_required
def dashboard(request):
    # Check if all authentication levels are complete
    session = AuthenticationSession.objects.filter(
        user=request.user,
        level_one_complete=True,
        level_two_complete=True,
        level_three_complete=True,
        session_key=request.session.session_key
    ).first()
    
    if not session:
        messages.error(request, 'Please complete all authentication levels.')
        return redirect('level_one')
    
    context = {
        'user': request.user,
        'authentication_complete': True
    }
    return render(request, 'dashboard.html', context)