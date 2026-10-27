# Lumen

A local chat interface for the existing knowledge graph. Users enter only a
question. The backend parses it, retrieves claims and original source passages,
and sends that context to Gemini. The answer returns to the conversation.

## Run

From `g_attention_rag/`, using the Python environment with the project's existing
dependencies and `en_core_web_sm` installed, put your key in the project `.env`
file (next to `main.py`):

```dotenv
GEMINI_API_KEY=your-api-key
GEMINI_MODEL=gemini-3.5-flash-lite
```

Then start the UI:

```powershell
python -B -m ui.server
```

Open **http://127.0.0.1:8000**. For another port, add `--port 8001`.
No Node build, frontend dependencies, or additional Python packages are needed.
The server automatically loads `.env`, which Git ignores. Existing process
environment values take precedence. Use one `NAME=value` setting per line;
matching quotes and full-line `#` comments are supported. Values are literal,
without variable expansion or inline comments. Restart after changing `.env`.
Alternatively, set the key for the current terminal using a hidden prompt:

```powershell
$geminiSecret = Read-Host "Gemini API key" -AsSecureString
$env:GEMINI_API_KEY = [System.Net.NetworkCredential]::new("", $geminiSecret).Password
python -B -m ui.server
```

The default model is `gemini-3.5-flash-lite`; set `GEMINI_MODEL` before starting to use
another model available to your API key. This uses Google's documented
[generateContent REST API](https://ai.google.dev/api/generate-content) and
[model identifiers](https://ai.google.dev/gemini-api/docs/models).

The server reads the project's existing `data/knowledge_graph.pkl` on the first
question. It does not rebuild or overwrite it. Restart the server after rebuilding
the graph. If the graph is missing, build one with the existing CLI, for example
`python main.py path/to/document.txt`.

## Modules

| File | Responsibility |
| --- | --- |
| `settings.py` | Paths, model, request limits, and server-side API key |
| `retrieval.py` | Existing query parser, graph anchor matching, and PathRAG retrieval |
| `prompt.py` | Question, claims, source passages, and grounding instructions |
| `gemini.py` | Gemini request, response parsing, and upstream errors |
| `server.py` | Local HTTP server, validation, and route orchestration |
| `static/index.html` | Chat layout |
| `static/styles.css` | Responsive visual design |
| `static/app.js` | Conversation interaction and browser storage |

Follow-up requests include the last six completed turns. Questions without new
entity matches inherit entities from the previous question; this is a simple
follow-up heuristic, not a separate LLM query rewriting step. Prompts include up
to 30 retrieved claims and 60,000 characters of source passages. Empty retrieval
asks Gemini to acknowledge the evidence gap instead of inventing an answer.

Up to 20 conversations are kept in this browser's local storage. They are not
stored by the server. Clearing this site's browser data removes them. Questions,
recent conversation context, and retrieved passages are sent to Google's Gemini
API when a question is submitted. No API key is exposed to the browser.

This server binds to localhost and is intended for local use. Responses are
displayed as plain text, including simple lists; HTML from documents or Gemini
is never executed. The interface waits for a complete response rather than
streaming tokens.

## Verification

From `g_attention_rag/`:

```powershell
python -B -m unittest discover -s tests -p test_ui.py -v
```

The tests use a simulated Gemini response, so they do not require an API key or
make paid requests. A real Gemini request still requires a valid key and quota.
