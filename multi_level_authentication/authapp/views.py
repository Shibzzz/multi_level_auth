from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from .models import UserProfile, AuthenticationSession
from .forms import UserRegistrationForm
from django.core.files.storage import FileSystemStorage
import random
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

def index(request):
    # If user is already authenticated, redirect to dashboard
    if request.user.is_authenticated:
        return redirect('dashboard')  # Make sure you have a 'dashboard' URL pattern
    # Otherwise show the index page
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
            
            # Use the correct URL name that matches urls.py
            return redirect('register_level_two')
        else:
            # Log form errors for debugging
            logger.error(f"Form validation errors: {form.errors}")
            messages.error(request, 'Please correct the errors below.')
            return render(request, 'register.html', {'form': form})
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
                return JsonResponse({'status': 'error', 'message': 'Registration data not found in session'})
            
            # Create user
            form = UserRegistrationForm(registration_data)
            if form.is_valid():
                user = form.save()
                
                # Read image data as binary
                image_data = image.read()
                
                # Create and save user profile with binary image data
                user_profile = UserProfile.objects.create(
                    user=user,
                    pattern_image=image_data,  # Store binary data directly
                    pattern=json.dumps(sorted_pattern)
                )
                
                # Clear the session data
                if 'registration_data' in request.session:
                    del request.session['registration_data']
                
                return JsonResponse({
                    'status': 'success',
                    'message': 'Pattern successfully registered!',
                    'redirect_url': '/register/level-three/'
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

def register_level_three(request):  
    return render(request, 'register_level_three.html')

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

@login_required
def level_three(request):
    if not request.session.get('level_two_complete'):
        return redirect('level_two')
    
    # Your level three authentication logic here
    return render(request, 'level_three.html')