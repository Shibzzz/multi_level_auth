"""
URL configuration for biometric_auth project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.contrib.auth import views as auth_views
from face_gesture import views as face_gesture_views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # Authentication flow
    path('', face_gesture_views.home, name='home'),
    path('login/', face_gesture_views.level_one, name='level_one'),
    path('level-two/', face_gesture_views.level_two, name='level_two'),
    path('level-two/complete/', face_gesture_views.level_two_complete, name='level_two_complete'),
    path('level-three/', face_gesture_views.level_three, name='level_three'),
    path('level-three/complete/', face_gesture_views.level_three_complete, name='level_three_complete'),
    path('dashboard/', face_gesture_views.dashboard, name='dashboard'),
    path('auth-status/', face_gesture_views.auth_status, name='auth_status'),
    path('logout/', auth_views.LogoutView.as_view(next_page='home'), name='logout'),
    
    # Pattern authentication
    path('pattern/auth/', face_gesture_views.pattern_authentication, name='pattern_authentication'),
    path('pattern/verify/', face_gesture_views.verify_pattern, name='verify_pattern'),
    path('pattern/image/', face_gesture_views.get_pattern_image, name='get_pattern_image'),
    
    # Registration flow
    path('register/', face_gesture_views.register, name='register'),
    path('register/level-two/', face_gesture_views.register_level_two, name='register_level_two'),
    path('register/level-three/', face_gesture_views.register_level_three, name='register_level_three'),
    path('register/level-three/face/api/', face_gesture_views.register_face_level_three_api, name='register_face_level_three_api'),
    path('register/level-three/gesture/', face_gesture_views.register_gesture_level_three, name='register_gesture_level_three'),
    path('register/level-three/gesture/api/', face_gesture_views.register_gesture_level_three_api, name='register_gesture_level_three_api'),
    path('register/complete/', face_gesture_views.register_complete, name='register_complete'),
    path('register/cancel/', face_gesture_views.register_cancel, name='register_cancel'),
    
    # Face authentication routes
    path('face/register/<str:user_id>/', face_gesture_views.register_face, name='register_face'),
    path('face/verify/<str:user_id>/', face_gesture_views.verify_face, name='verify_face'),
    
    # Gesture authentication routes
    path('gesture/register/', face_gesture_views.register_gesture, name='register_gesture'),
    path('gesture/verify/', face_gesture_views.verify_gesture, name='verify_gesture'),
    
    # Direct gesture recognition routes
    path('gesture/register-direct/<str:user_id>/', face_gesture_views.register_gesture_direct, name='register_gesture_direct'),
    path('gesture/verify-direct/<str:user_id>/', face_gesture_views.verify_gesture_direct, name='verify_gesture_direct'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
