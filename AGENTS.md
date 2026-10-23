# Repository Guidelines

## Project Structure & Module Organization

`g_attention_rag/main.py` runs the HotpotQA knowledge graph pipeline. Implemented modules in `g_attention_rag/src/` include `query_parser.py` and `graph_builder.py`; traversal, attention pruning, and generation modules are placeholders. The query parser is currently disconnected from the main CLI. `config.py` and `eval/` also contain placeholders.

Both root-level `data/` and `g_attention_rag/data/` contain sample inputs and generated artifacts. Paths resolve relative to the working directory; run commands from `g_attention_rag/` consistently. `References/`, `Research Papers/`, and `Synopsis/` hold research and project documents.

## Build, Test, and Development Commands

From the repository root, create and activate a virtual environment in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
cd g_attention_rag
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
python main.py --sample-idx 0
python main.py --sample-idx 0 --all-context
```

The first run builds a graph from supporting documents; `--all-context` includes distractor paragraphs. Runs write `data/knowledge_graph.pkl`, `data/processed_graph/graph_output.json`, and a graph PNG. Missing sample caches trigger a Hugging Face dataset download. There is no separate build step.

## Coding Style & Naming Conventions

Follow existing Python conventions: four-space indentation, `snake_case` functions and variables, `PascalCase` classes, and uppercase constants. Retain type hints and concise docstrings for public functions. Keep graph construction and query parsing in their respective modules. Keep entity canonicalization consistent with future anchor lookup logic. No formatter or linter configuration is currently provided.

## Testing Guidelines

No automated test suite, testing framework, or coverage threshold is configured. Smoke-test graph changes with both CLI commands above; inspect node counts, directed relations, JSON output, and the rendered graph. For extraction changes, check active/passive sentences, coordinated entities, and co-occurrence fallbacks. Name future automated tests `test_*.py` under `g_attention_rag/tests/` and document their runner.

## Commit & Pull Request Guidelines

History uses short descriptive messages without a consistent enforced format. Use imperative, specific summaries such as `Fix passive-voice graph extraction`. Keep changes focused. PRs should describe behavior changes, validation commands and results, and relevant issues. Include before/after graph images when visualization changes. Review generated data diffs before committing; avoid adding virtual environments or bytecode caches.
