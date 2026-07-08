"""Client-side web tools for flow.agent — Exa search + page reader.

Anthropic's server-side web_search only exists on the Anthropic API.
On any other transport (OpenRouter, local models), agents need
client-side tools: the model calls them, we execute, the result goes
back as a tool_result segment. That restores true iterative research
loops — search, read, refine, search again — on models that cost
pennies.

Requires EXA_API_KEY in the environment (.env works — motif loads it).
Search costs ~$0.005-0.007, contents ~$0.001/page. A free keyless
alternative for read_page is Jina's reader (https://r.jina.ai/<url>,
~20 RPM) if Exa credits run dry.

    from websearch import WEB_TOOLS, WEB_TOOL_SCHEMAS
    result = await flow.agent(msg, tools=WEB_TOOLS,
                              tool_schemas=WEB_TOOL_SCHEMAS, max_steps=8)
"""

import os

import httpx

_EXA = "https://api.exa.ai"

_http: httpx.AsyncClient | None = None


def _client() -> httpx.AsyncClient:
    global _http
    if _http is None:
        _http = httpx.AsyncClient(
            timeout=60.0,
            headers={"x-api-key": os.environ["EXA_API_KEY"],
                     "Content-Type": "application/json"},
        )
    return _http


async def web_search(input: dict) -> str:
    """Search the web via Exa. Returns titles, URLs, and snippets."""
    query = input["query"]
    n = min(int(input.get("num_results", 5)), 10)
    resp = await _client().post(f"{_EXA}/search", json={
        "query": query,
        "numResults": n,
        "contents": {"text": {"maxCharacters": 500}},
    })
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        return f"No results for: {query}"
    parts = []
    for r in results:
        parts.append(
            f"## {r.get('title') or '(untitled)'}\n"
            f"URL: {r.get('url')}\n"
            f"Published: {r.get('publishedDate', 'unknown')}\n\n"
            f"{(r.get('text') or '').strip()}"
        )
    return "\n\n---\n\n".join(parts)


async def read_page(input: dict) -> str:
    """Fetch a page's text content via Exa contents."""
    url = input["url"]
    resp = await _client().post(f"{_EXA}/contents", json={
        "urls": [url],
        "text": {"maxCharacters": 8000},
    })
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results or not (results[0].get("text") or "").strip():
        return f"No readable content at {url}"
    r = results[0]
    return f"# {r.get('title') or url}\n\n{r['text'].strip()}"


WEB_TOOLS = {
    "web_search": web_search,
    "read_page": read_page,
}

# Anthropic-format schemas — llm.act() converts on the wire for
# OpenAI-compatible transports.
WEB_TOOL_SCHEMAS = [
    {
        "name": "web_search",
        "description": (
            "Search the web. Returns titles, URLs, publication dates, and "
            "text snippets. Refine and search again if results are thin."),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "num_results": {"type": "integer",
                                "description": "How many results (max 10, default 5)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_page",
        "description": (
            "Fetch the full text of a specific page by URL. Use after "
            "web_search to read a promising source in depth."),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to read"},
            },
            "required": ["url"],
        },
    },
]
