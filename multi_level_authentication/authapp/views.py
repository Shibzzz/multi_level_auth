from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from .models import UserProfile, AuthenticationSession
from .forms import UserRegistrationForm
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
        # Store level 1 data in session
        request.session['registration_data'] = {
            'username': request.POST.get('username'),
            'email': request.POST.get('email'),
            'password1': request.POST.get('password1'),
            'password2': request.POST.get('password2'),
        }
        return redirect('register_level_two')
    else:
        form = UserRegistrationForm()
    
    return render(request, 'register.html', {'form': form})

def register_level_two(request):
    if request.method == 'POST':
        pattern = json.loads(request.POST.get('pattern'))
        
        # Sort pattern by placement time to show actual placement order
        sorted_pattern = sorted(pattern, key=lambda x: int(x.get('placementTime', 0)))
        
        logger.info("\n=== Pattern Registration Details ===")
        logger.info(f"Registration Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("Pattern Sequence (in order of placement):")
        
        for index, piece in enumerate(sorted_pattern, 1):
            logger.info(f"Step {index}: Grid piece {piece['gridPosition']} → Position {piece['targetPosition']}")
        
        logger.info("=" * 35 + "\n")
        
        # Get the uploaded image
        if 'image' not in request.FILES:
            return JsonResponse({'status': 'error', 'message': 'No image uploaded'})
            
        image = request.FILES['image']
        dropped_positions = request.POST.getlist('dropped_positions[]')
        validation_points = request.POST.getlist('validation_points[]')
        
        # Validate inputs
        if len(dropped_positions) != 2:
            return JsonResponse({'status': 'error', 'message': 'Please select exactly 2 grid positions'})
        
        try:
            # Save the image
            fs = FileSystemStorage()
            filename = fs.save(f'pattern_images/{request.user.id}_{image.name}', image)
            
            # Save pattern positions
            user_profile = UserProfile.objects.get_or_create(user=request.user)[0]
            user_profile.pattern_image = filename
            user_profile.pattern_position_1 = int(dropped_positions[0])
            user_profile.pattern_position_2 = int(dropped_positions[1])
            user_profile.save()
            
            return JsonResponse({
                'status': 'success',
                'message': 'Pattern successfully registered!',
                'redirect_url': '/register/level-three/'  # Adjust this URL as needed
            })
            
        except Exception as e:
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

@login_required
def level_three(request):
    if not request.session.get('level_two_complete'):
        return redirect('level_two')
    
    # Your level three authentication logic here
    return render(request, 'level_three.html')