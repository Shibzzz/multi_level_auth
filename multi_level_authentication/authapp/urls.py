from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.index, name='index'),
    
    # Registration paths
    path('register/', views.register, name='register'),
    path('register/level-two/', views.register_level_two, name='register_level_two'),
    path('register/level-three/', views.register_level_three, name='register_level_three'),
    
    # Login/authentication paths
    path('level-one/', views.level_one, name='level_one'),
    path('level-two/', views.level_two, name='level_two'),
    path('verify-level-three/', views.verify_level_three, name='verify_level_three'),
    
    # Dashboard
    path('dashboard/', views.dashboard, name='dashboard'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)