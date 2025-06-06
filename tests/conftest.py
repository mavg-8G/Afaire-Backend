import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from main import app

@pytest.fixture(autouse=True)
def disable_rate_limiting():
    class DummyLimiter:
        def limit(self, *args, **kwargs):
            def decorator(f):
                return f
            return decorator
        def _inject_headers(self, response, view_rate_limit):
            return response
    app.state.limiter = DummyLimiter()
    yield
