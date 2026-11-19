import os
import sys
from urllib.parse import urlparse

# Ensure root directory is on sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app import app

class VercelPathFixMiddleware:
    """
    Vercel CLI 65+ rewrites incoming requests to the destination path (/api/index).
    This middleware restores the original requested path from Vercel's HTTP_X_MATCHED_PATH
    or other forwarded headers, and cleanly resolves any /api/index prefix, ensuring
    Flask routes (/, /login, /dashboard, /static/...) match accurately.
    """
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        # 1. Check HTTP_X_MATCHED_PATH (Vercel standard for rewritten URLs)
        matched_path = environ.get('HTTP_X_MATCHED_PATH')
        if not matched_path:
            matched_path = environ.get('HTTP_X_FORWARDED_URI') or environ.get('HTTP_X_NOW_ROUTE_MATCHES')

        if matched_path:
            parsed = urlparse(matched_path)
            environ['PATH_INFO'] = parsed.path
            if parsed.query and not environ.get('QUERY_STRING'):
                environ['QUERY_STRING'] = parsed.query
        else:
            path_info = environ.get('PATH_INFO', '')
            if path_info in ('/api/index', '/api/index/'):
                environ['PATH_INFO'] = '/'
            elif path_info.startswith('/api/index/'):
                environ['PATH_INFO'] = path_info[len('/api/index'):]

        return self.wsgi_app(environ, start_response)

app.wsgi_app = VercelPathFixMiddleware(app.wsgi_app)
