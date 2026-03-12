"""
Local fallback for flask_cors used when external packages are unavailable.
Supports `CORS(app)` with permissive defaults.
"""


def CORS(app):
    @app.after_request
    def _add_cors_headers(response):
        response.headers.setdefault("Access-Control-Allow-Origin", "*")
        response.headers.setdefault("Access-Control-Allow-Headers", "*")
        response.headers.setdefault("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS")
        return response

    return app
