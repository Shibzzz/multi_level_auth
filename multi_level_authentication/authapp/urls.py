from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.index, name='index'),

    path('register/', views.register, name='register'),
    path('register_level_two/', views.register_level_two, name='register_level_two'),
    path('level_one/', views.level_one, name='level_one'),  # Changed to 'login/'
    path('level-two/', views.level_two, name='level_two'),
    path('level-three/', views.level_three, name='level_three'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)