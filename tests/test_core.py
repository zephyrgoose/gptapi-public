import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Provide a tiny YAML stub so gptapi_core imports without PyYAML installed
yaml_stub = types.SimpleNamespace()
def _safe_load(data):
    if hasattr(data, 'read'):
        data = data.read()
    d = {}
    for line in str(data).splitlines():
        if ':' in line:
            k, v = line.split(':', 1)
            val = v.strip()
            if val.isdigit():
                val = int(val)
            d[k.strip()] = val
    return d
yaml_stub.safe_load = _safe_load
sys.modules.setdefault('yaml', yaml_stub)

# Minimal stub for openai.OpenAI used in gptapi_core
class _DummyOpenAI:
    def __init__(self, *a, **k):
        pass

    class chat:
        class completions:
            @staticmethod
            def create(**kwargs):
                class Resp:
                    choices = [types.SimpleNamespace(message=types.SimpleNamespace(tool_calls=[types.SimpleNamespace(function=types.SimpleNamespace(arguments='{}'))]))]
                return Resp()

    class responses:
        @staticmethod
        def create(**kwargs):
            class Resp:
                output_text = 'result'
                content = []
            return Resp()

sys.modules.setdefault('openai', types.SimpleNamespace(OpenAI=_DummyOpenAI))

from gptapi_core import _load_yaml, load_profile, gptapi

def test_load_yaml(tmp_path):
    p = tmp_path / "sample.yaml"
    p.write_text("a: 1\n")
    assert _load_yaml(p) == {"a": 1}

def test_load_profile():
    base = os.path.join(os.path.dirname(__file__), "..", "profiles")
    prof = load_profile("reason", base)
    for key in ("model", "system_prompt", "parameters", "structured_output"):
        assert key in prof

def test_gptapi_calls(monkeypatch):
    called = {}
    def fake_call(profile, messages):
        called['profile'] = profile
        called['messages'] = messages
        return {'ok': True}

    monkeypatch.setattr('gptapi_core.call_structured', fake_call)
    result = gptapi("reason", "hi", base_dir=os.path.join(os.path.dirname(__file__), "..", "profiles"))
    assert result == {'ok': True}
    assert called['messages'][0]['content'] == "hi"
