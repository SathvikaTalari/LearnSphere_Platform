import os
import sys
from urllib.parse import parse_qs, urlencode, urlparse

# Ensure root directory is on sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app import app

class VercelPathFixMiddleware:
    """
    Restores the real URL path when running behind Vercel serverless rewrites.
    
    1. Reads __path__ passed by vercel.json rewrite (most reliable).
    2. Falls back to HTTP_X_MATCHED_PATH / HTTP_X_FORWARDED_URI if present.
    3. Normalizes and cleans PATH_INFO and QUERY_STRING for Flask.
    """
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        qs = environ.get('QUERY_STRING', '')
        
        # Priority 1: Check __path__ injected by vercel.json rewrite
        if '__path__' in qs:
            params = parse_qs(qs, keep_blank_values=True)
            if '__path__' in params:
                raw_target = params.pop('__path__')[0]
                clean = '/' + raw_target.lstrip('/')
                environ['PATH_INFO'] = clean
                environ['QUERY_STRING'] = urlencode(params, doseq=True)
                return self.wsgi_app(environ, start_response)

        # Priority 2: Check HTTP_X_MATCHED_PATH / HTTP_X_FORWARDED_URI
        matched = environ.get('HTTP_X_MATCHED_PATH') or environ.get('HTTP_X_FORWARDED_URI') or ''
        if matched and not matched.startswith('/api/index'):
            parsed = urlparse(matched)
            environ['PATH_INFO'] = '/' + parsed.path.lstrip('/')
            return self.wsgi_app(environ, start_response)

        # Priority 3: If PATH_INFO is literally /api/index, strip it or default to /
        path_info = environ.get('PATH_INFO', '')
        if path_info in ('/api/index', '/api/index/'):
            environ['PATH_INFO'] = '/'
        elif path_info.startswith('/api/index/'):
            environ['PATH_INFO'] = '/' + path_info[len('/api/index'):].lstrip('/')

        return self.wsgi_app(environ, start_response)

app.wsgi_app = VercelPathFixMiddleware(app.wsgi_app)
