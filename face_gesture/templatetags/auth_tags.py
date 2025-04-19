from django import template
from django.urls import reverse
from face_gesture.models import AuthenticationSession

register = template.Library()

@register.simple_tag(takes_context=True)
def get_next_auth_url(context):
    """
    Return the URL for the next authentication step based on current user's progress.
    Used to determine where to redirect after login or successful level completion.
    """
    request = context['request']
    if not request.user.is_authenticated:
        return reverse('level_one')
    
    auth_session = AuthenticationSession.objects.filter(
        user=request.user,
        session_key=request.session.session_key
    ).first()
    
    if not auth_session:
        # If no auth session exists, start with level one
        return reverse('level_one')
    
    if not auth_session.level_one_complete:
        return reverse('level_one')
    elif not auth_session.level_two_complete:
        return reverse('level_two')
    elif not auth_session.level_three_complete:
        return reverse('level_three')
    else:
        # All levels complete, redirect to dashboard or home
        return reverse('dashboard')

@register.filter
def auth_level_name(level):
    """Convert numeric authentication level to readable name."""
    level_names = {
        0: "Not Authenticated",
        1: "Level One (Password)",
        2: "Level Two (Pattern)",
        3: "Level Three (Biometric)"
    }
    return level_names.get(level, "Unknown")

@register.simple_tag(takes_context=True)
def user_needs_auth(context):
    """Check if the user needs to complete any authentication level."""
    request = context['request']
    if not request.user.is_authenticated:
        return True
    
    auth_session = AuthenticationSession.objects.filter(
        user=request.user,
        session_key=request.session.session_key
    ).first()
    
    if not auth_session:
        return True
    
    # Check if all three levels are complete
    return not (auth_session.level_one_complete and 
                auth_session.level_two_complete and 
                auth_session.level_three_complete)

@register.inclusion_tag('face_gesture/tags/auth_progress.html', takes_context=True)
def auth_progress(context):
    """
    Render an authentication progress component showing all levels
    and their completion status.
    """
    request = context['request']
    current_level = 0
    level_one_done = False
    level_two_done = False
    level_three_done = False
    
    if request.user.is_authenticated:
        auth_session = AuthenticationSession.objects.filter(
            user=request.user,
            session_key=request.session.session_key
        ).first()
        
        if auth_session:
            level_one_done = auth_session.level_one_complete
            level_two_done = auth_session.level_two_complete
            level_three_done = auth_session.level_three_complete
            
            if level_three_done:
                current_level = 3
            elif level_two_done:
                current_level = 2
            elif level_one_done:
                current_level = 1
    
    return {
        'current_level': current_level,
        'level_one_done': level_one_done,
        'level_two_done': level_two_done,
        'level_three_done': level_three_done,
        'level_percent': current_level * 33.33,
        'request': request,
    } 