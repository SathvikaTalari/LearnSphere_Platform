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
    This middleware restores the true requested path from Vercel's HTTP_X_MATCHED_PATH
    or forwarded headers, and cleanly strips any /api/index prefix, ensuring
    Flask routes (/, /login, /student/dashboard, /api/...) match accurately.
    """
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        raw_path = (
            environ.get('HTTP_X_MATCHED_PATH') or
            environ.get('HTTP_X_FORWARDED_URI') or
            environ.get('HTTP_X_NOW_ROUTE_MATCHES') or
            environ.get('PATH_INFO', '')
        )

        parsed = urlparse(raw_path)
        clean_path = parsed.path or '/'
        if clean_path in ('/api/index', '/api/index/'):
            clean_path = '/'
        elif clean_path.startswith('/api/index/'):
            clean_path = clean_path[len('/api/index'):] or '/'

        environ['PATH_INFO'] = clean_path
        if parsed.query and not environ.get('QUERY_STRING'):
            environ['QUERY_STRING'] = parsed.query

        return self.wsgi_app(environ, start_response)

app.wsgi_app = VercelPathFixMiddleware(app.wsgi_app)
