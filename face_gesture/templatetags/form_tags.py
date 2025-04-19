from django import template
from face_gesture.models import AuthenticationSession

register = template.Library()

@register.filter(name='add_class')
def add_class(field, css_class):
    """
    Adds a CSS class to the specified form field
    Usage: {{ form.field|add_class:"form-control" }}
    """
    return field.as_widget(attrs={"class": css_class})

@register.filter(name='mul')
def multiply(value, arg):
    """Multiply the value by the argument."""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return 0

@register.simple_tag(takes_context=True)
def get_auth_level(context):
    """Return the current authentication level for the user in this session."""
    request = context['request']
    if not request.user.is_authenticated:
        return 0
    
    auth_session = AuthenticationSession.objects.filter(
        user=request.user,
        session_key=request.session.session_key
    ).first()
    
    if not auth_session:
        return 0
    
    if auth_session.level_three_complete:
        return 3
    elif auth_session.level_two_complete:
        return 2
    elif auth_session.level_one_complete:
        return 1
    else:
        return 0

@register.simple_tag(takes_context=True)
def has_completed_level(context, level):
    """Check if user has completed a specific authentication level."""
    request = context['request']
    if not request.user.is_authenticated:
        return False
    
    auth_session = AuthenticationSession.objects.filter(
        user=request.user,
        session_key=request.session.session_key
    ).first()
    
    if not auth_session:
        return False
    
    if level == 1:
        return auth_session.level_one_complete
    elif level == 2:
        return auth_session.level_two_complete
    elif level == 3:
        return auth_session.level_three_complete
    
    return False 