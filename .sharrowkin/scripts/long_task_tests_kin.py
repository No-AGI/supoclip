"""Isolated container test entry point; mounted sources only, never dotenv."""
import os
import sys
from pathlib import Path

os.environ.clear()
os.environ.update({"PATH": "/app/.venv/bin:/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp", "PYTHONDONTWRITEBYTECODE": "1", "DATABASE_URL": "", "TEST_DATABASE_URL": "", "SELF_HOST": "true", "TEMP_DIR": "/tmp/kin-media", "LLM": "ollama:test", "AWS_EC2_METADATA_DISABLED": "true"})
import dotenv
dotenv.load_dotenv = lambda *args, **kwargs: False
sys.path.insert(0, "/check")
os.chdir("/tmp")
# Prefer mounted tests (namespace package) over the image's older test package.
import types
package = types.ModuleType("tests")
package.__path__ = ["/check/tests"]
sys.modules["tests"] = package
unit = types.ModuleType("tests.unit")
unit.__path__ = ["/check/tests/unit"]
sys.modules["tests.unit"] = unit
import pytest
raise SystemExit(pytest.main(["-c", "/check/pyproject.toml", "--confcutdir=/check/tests/unit", "--no-cov", "-p", "no:cacheprovider", "-q", *["/check/tests/unit/" + name for name in sys.argv[1:]]]))
