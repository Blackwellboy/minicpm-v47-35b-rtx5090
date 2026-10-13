import json, sys, time, urllib.request
B = "http://127.0.0.1:18111/v1"; M = "MiniCPM-V-4.7-35B-A3B-AWQ"
def post(body):
    body = {"model": M, **body}
    r = urllib.request.Request(B + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.perf_counter(); p = json.loads(urllib.request.urlopen(r, timeout=600).read()); return p, time.perf_counter() - t
out = {}
p, dt = post({"messages": [{"role": "user", "content": "What is 17*23? Reply with the number only."}], "max_tokens": 32, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
m = p["choices"][0]["message"]; out["think_off"] = {"content": m.get("content"), "reasoning": (m.get("reasoning") or m.get("reasoning_content")), "finish": p["choices"][0]["finish_reason"], "s": round(dt, 2)}
p, dt = post({"messages": [{"role": "user", "content": "A bat and ball cost $1.10 total; the bat costs $1 more than the ball. Ball price in cents? Final answer as a number."}], "max_tokens": 4096, "temperature": 1.0, "top_p": 0.95, "top_k": 20})
m = p["choices"][0]["message"]; r = m.get("reasoning") or m.get("reasoning_content") or ""
out["think_on"] = {"content": (m.get("content") or "")[-200:], "reasoning_chars": len(r), "think_tag_leak": "<think>" in (m.get("content") or "") or "</think>" in (m.get("content") or ""), "usage": p["usage"], "finish": p["choices"][0]["finish_reason"], "s": round(dt, 2), "decode_tps": round(p["usage"]["completion_tokens"] / dt, 1)}
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}}]
p, dt = post({"messages": [{"role": "user", "content": "What's the weather in Melbourne in celsius?"}], "tools": tools, "max_tokens": 512, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
m = p["choices"][0]["message"]; out["tool"] = {"tool_calls": m.get("tool_calls"), "content": m.get("content"), "finish": p["choices"][0]["finish_reason"]}
print(json.dumps(out, indent=1))
