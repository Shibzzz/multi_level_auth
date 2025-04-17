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
from face_gesture import views as face_gesture_views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    # Face authentication routes
    path('face/register/<str:user_id>/', face_gesture_views.register_face, name='register_face'),
    path('face/verify/<str:user_id>/', face_gesture_views.verify_face, name='verify_face'),
    # Gesture authentication routes
    path('gesture/register/', face_gesture_views.register_gesture, name='register_gesture'),
    path('gesture/verify/', face_gesture_views.verify_gesture, name='verify_gesture'),
    # New direct gesture recognition routes
    path('gesture/register-direct/<str:user_id>/', face_gesture_views.register_gesture_direct, name='register_gesture_direct'),
    path('gesture/verify-direct/<str:user_id>/', face_gesture_views.verify_gesture_direct, name='verify_gesture_direct'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
