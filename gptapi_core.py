# gptapi_core.py
# Written by Dan W + GPT4

import json, os, yaml
from openai import OpenAI

def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_profile(name, base_dir):
    # Tries .yaml, .yml, and .json for profile config, in that order.
    for ext in (".yaml", ".yml", ".json"):
        path = os.path.join(base_dir, f"{name}{ext}")
        if os.path.exists(path):
            break
    else:
        # No config file found.
        raise ValueError(f"profile '{name}' not found (yaml/json)")
    # Load YAML or JSON depending on file extension.
    with open(path, "r", encoding="utf-8") as f:
        prof = json.load(f) if path.endswith(".json") else yaml.safe_load(f)
    # All profiles must specify these keys:
    for k in ("model", "system_prompt", "parameters", "structured_output"):
        if k not in prof:
            raise ValueError(f"profile '{name}' missing '{k}'")
    return prof

def _load_key(path):
    keys = _load_yaml(path)
    return keys["openai_api"]

def call_structured(profile: dict, messages: list[dict]):
    so = profile["structured_output"]
    fn = {
      "type": "function",
      "function": {
        "name": so["name"],
        "description": "Strict JSON output per schema",
        "parameters": so["schema"]
      }
    }
    client = OpenAI(api_key=_load_key(profile.get("credentials_file", "./keys.yaml")))
    resp = client.chat.completions.create(
        model       = profile["model"],
        messages    = [{"role":"system","content":profile["system_prompt"]}, *messages],
        tools       = [fn],
        tool_choice = {"type":"function", "function":{"name":so["name"]}},
        **profile["parameters"]
    ).choices[0].message

    # extract JSON
    if getattr(resp, "tool_calls", None):
        out = json.loads(resp.tool_calls[0].function.arguments)
    elif getattr(resp, "function_call", None):
        out = json.loads(resp.function_call.arguments)
    else:
        raise RuntimeError("No structured output returned")

    # if model used "result" instead of userRequestedOutput, remap it
    if "userRequestedOutput" not in out and "result" in out:
        out["userRequestedOutput"] = out.pop("result")

    return out

# ── Web-search wrapper ────────────────────────────────────────────────────
def run_web_search(query: str, *, model: str = "gpt-4.1-mini") -> str:
    """
    Execute a live OpenAI web-search tool call and return result summary
    with explicit references/citations included, at the end. 

    This constructs references using the annotations in the response.
    """
    client = OpenAI(api_key=_load_key("./keys.yaml"))

    resp = client.responses.create(
        model=model,
        tools=[{
            "type": "web_search_preview",
            "search_context_size": "high",
            "user_location": {
                "type": "approximate",
                "country": "AU",
                "city": "Melbourne",
                "region": "Victoria",
            }
        }],
        input=query
    )

    # Defensive: try both 'output_text' and detailed structures for safety
    if hasattr(resp, "output_text") and getattr(resp, "output_text"):
        # Fallback, if model just returns string
        return resp.output_text.strip()

    # Prefer structured output if present
    # According to docs, resp content is in resp.content (list)
    if not hasattr(resp, "content"):
        raise RuntimeError("web_search_preview: response lacks .content field")

    segments = []
    references = []
    ref_counter = 1
    url_to_refnum = {}  # so the same URL gets same reference number

    for content_item in resp.content:
        if getattr(content_item, "type", None) == "output_text":
            txt = getattr(content_item, "text", "")
            # Add inline footnotes to text for URLs
            annotations = getattr(content_item, "annotations", [])
            annotated_text = txt
            # Build up replacements (in descending index order!)
            replace_points = []
            for ann in annotations:
                if getattr(ann, "type", "") == "url_citation":
                    url = getattr(ann, "url", None)
                    title = getattr(ann, "title", None)
                    start = getattr(ann, "start_index", None)
                    end = getattr(ann, "end_index", None)
                    if url is None or start is None or end is None:
                        continue
                    if url not in url_to_refnum:
                        url_to_refnum[url] = ref_counter
                        references.append((ref_counter, url, title))
                        ref_counter += 1
                    refnum = url_to_refnum[url]
                    # Insert [refnum] after end_index
                    replace_points.append( (end, f"[{refnum}]") )
            # Perform replacements from back to front so indexes still work
            annotated_text_parts = []
            last_index = 0
            sorted_points = sorted(replace_points, reverse=False)
            for insert_at, marker in sorted_points:
                annotated_text_parts.append(txt[last_index:insert_at])
                annotated_text_parts.append(marker)
                last_index = insert_at
            annotated_text_parts.append(txt[last_index:])
            annotated_text = "".join(annotated_text_parts)
            segments.append(annotated_text)

    references_section = ""
    if references:
        references_section = "\n\nReferences:\n"
        for refnum, url, title in references:
            references_section += f"[{refnum}]: {title or url} <{url}>\n"

    return "\n".join(segments).strip() + references_section
