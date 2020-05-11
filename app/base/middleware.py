from importlib import import_module

from django.conf import settings
from django.urls import resolve
from django.utils.deprecation import MiddlewareMixin


class SessionKeyMiddleware(MiddlewareMixin):
    def __init__(self, get_response=None):
        super().__init__(get_response)
        self.get_response = get_response
        engine = import_module(settings.SESSION_ENGINE)
        self.SessionStore = engine.SessionStore

    def process_request(self, request):
        all_apps = resolve(request.path_info).app_names
        auth_apps = settings.API_SESSION_KEY_APPS

        if any([app_name in all_apps for app_name in auth_apps]):
            params = None
            if request.method == "GET":
                params = request.GET
            elif request.method == "POST":
                params = request.POST

            if params is not None:
                session_key = params.get(settings.API_SESSION_KEY_QUERY_PARAM_NAME)
                if session_key is not None:
                    request.session = self.SessionStore(session_key)
