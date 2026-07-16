"""A fake codex CLI for testing the delegated adapter at the real
subprocess boundary. Scenario chosen by FAKE_CODEX_SCENARIO:

    ok           normal run: search item + answer + usage
    schema       structured answer honoring --output-schema presence
    schemareject rejects --output-schema like OpenAI strict mode
                 (turn.failed + invalid_json_schema, exit 1); answers
                 normally when invoked without it — the ladder test
    hang         never answers (deadline test)
    autherr      exit 2 with an auth message on stderr
    ratelimit    exit 1 with a rate limit message
    noanswer     clean exit, no agent_message
    garbage      valid answer surrounded by non-JSON noise lines

Also asserts (by crashing loudly) that the hermetic flags the adapter
promises are actually present in argv, and records argv + stdin to
FAKE_CODEX_LOG if set — the tests check what really crossed the
process boundary.
"""

import json
import os
import sys
import time

argv = sys.argv[1:]

if argv == ["--version"]:
    print("fake-codex 0.0.1")
    sys.exit(0)

scenario = os.environ.get("FAKE_CODEX_SCENARIO", "ok")

# The adapter's contract: hermetic flags always present, prompt on stdin.
required = ["exec", "--ephemeral", "--ignore-user-config", "-s", "read-only",
            "--json", "mcp_servers={}"]
joined = " ".join(argv)
for token in required:
    assert token in argv or token in joined, f"missing hermetic flag: {token}"
assert argv[-1] == "-", "prompt must arrive via stdin ('-')"

prompt = sys.stdin.read()

log_path = os.environ.get("FAKE_CODEX_LOG")
if log_path:
    with open(log_path, "w") as f:
        json.dump({"argv": argv, "prompt": prompt, "cwd": os.getcwd()}, f)


def emit(obj):
    print(json.dumps(obj), flush=True)


if scenario == "hang":
    time.sleep(120)
    sys.exit(0)

if scenario == "autherr":
    print("Error: not logged in — run codex login", file=sys.stderr)
    sys.exit(2)

if scenario == "ratelimit":
    print("Error: rate limit exceeded, retry later", file=sys.stderr)
    sys.exit(1)

emit({"type": "thread.started", "thread_id": "t1"})
emit({"type": "turn.started"})

if scenario == "schemareject":
    if "--output-schema" in argv:
        emit({"type": "error",
              "message": '{"error": {"code": "invalid_json_schema"}}'})
        emit({"type": "turn.failed",
              "error": {"message": '{"code": "invalid_json_schema", '
                                   '"param": "text.format.schema"}'}})
        sys.exit(1)
    # rung 2: schema arrived embedded in the prompt instead
    assert "matching this schema" in prompt
    emit({"type": "item.completed",
          "item": {"id": "m1", "type": "agent_message",
                   "text": json.dumps({"answer": 43})}})
    emit({"type": "turn.completed",
          "usage": {"input_tokens": 900, "cached_input_tokens": 0,
                    "output_tokens": 30, "reasoning_output_tokens": 5}})
    sys.exit(0)

if scenario == "garbage":
    print("[2026-07-16] diagnostic noise, not json", flush=True)

if scenario != "noanswer":
    emit({"type": "item.completed",
          "item": {"id": "s1", "type": "web_search", "query": "test query"}})
    schema_requested = "--output-schema" in argv
    if scenario == "schema" and schema_requested:
        text = json.dumps({"answer": 42, "echo": prompt[:40]})
    elif scenario == "schema":
        text = json.dumps({"error": "schema file never reached argv"})
    else:
        text = f"fake answer to: {prompt[:60]}"
    emit({"type": "item.completed",
          "item": {"id": "m1", "type": "agent_message", "text": text}})

emit({"type": "turn.completed",
      "usage": {"input_tokens": 1000, "cached_input_tokens": 400,
                "output_tokens": 50, "reasoning_output_tokens": 10}})
sys.exit(0)
