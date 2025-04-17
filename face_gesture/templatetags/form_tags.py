from django import template

register = template.Library()

@register.filter(name='add_class')
def add_class(field, css_class):
    """Add a CSS class to a form field."""
    attrs = field.field.widget.attrs
    
    if 'class' in attrs:
        attrs['class'] += f' {css_class}'
    else:
        attrs['class'] = css_class
        
    return field 