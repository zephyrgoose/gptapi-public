# gptapi_core.py
# Written by Dan W + GPT4

import json, os, yaml, asyncio
from openai import AsyncOpenAI
from pydantic import BaseModel


class ProfileModel(BaseModel):
    model: str
    system_prompt: str
    parameters: dict
    structured_output: dict
    credentials_file: str | None = "./keys.yaml"


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    try:
        return _load_yaml("./keys.yaml")["openai_api"]
    except Exception:
        return None


client = AsyncOpenAI(api_key=_resolve_api_key())

def load_profile(name, base_dir):
    """Load a YAML or JSON profile and validate it."""
    for ext in (".yaml", ".yml", ".json"):
        path = os.path.join(base_dir, f"{name}{ext}")
        if os.path.exists(path):
            break
    else:
        raise ValueError(f"profile '{name}' not found (yaml/json)")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f) if path.endswith(".json") else yaml.safe_load(f)

    return ProfileModel(**data).model_dump()

async def call_structured_async(profile: dict, messages: list[dict]):
    so = profile["structured_output"]
    fn = {
      "type": "function",
      "function": {
        "name": so["name"],
        "description": "Strict JSON output per schema",
        "parameters": so["schema"]
      }
    }
    resp = await client.chat.completions.create(
        model       = profile["model"],
        messages    = [{"role":"system","content":profile["system_prompt"]}, *messages],
        tools       = [fn],
        tool_choice = {"type":"function", "function":{"name":so["name"]}},
        **profile["parameters"]
    )
    resp_msg = resp.choices[0].message

    # extract JSON
    if getattr(resp_msg, "tool_calls", None):
        out = json.loads(resp_msg.tool_calls[0].function.arguments)
    elif getattr(resp_msg, "function_call", None):
        out = json.loads(resp_msg.function_call.arguments)
    else:
        raise RuntimeError("No structured output returned")

    # if model used "result" instead of userRequestedOutput, remap it
    if "userRequestedOutput" not in out and "result" in out:
        out["userRequestedOutput"] = out.pop("result")

    return out

def call_structured(profile: dict, messages: list[dict]):
    """Synchronous wrapper around :func:`call_structured_async`."""
    return asyncio.run(call_structured_async(profile, messages))


def gptapi(profile_name: str, prompt: str, base_dir: str | None = None):
    """Convenience wrapper to load a profile and call the model."""
    base_dir = base_dir or os.path.join(os.path.dirname(__file__), "profiles")
    profile = load_profile(profile_name, base_dir)
    messages = [{"role": "user", "content": prompt}]
    return call_structured(profile, messages)
