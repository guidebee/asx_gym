"""
WSGI config for asx_data project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/2.2/howto/deployment/wsgi/
"""

import os
import sys
sys.path.append('/var/www/axs_data/')
sys.path.append('/anaconda3/lib/python3.7/site-packages/')


from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'asx_data.settings')

application = get_wsgi_application()
