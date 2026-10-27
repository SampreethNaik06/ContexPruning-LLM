# ContexPruning-LLM (ECAP-RAG)

Entity-Context-Aware Pruning for Retrieval-Augmented Generation (ECAP-RAG). This project extracts Subject-Verb-Object (SVO) claims and entity networks from documents into a traversable knowledge graph, prunes irrelevant context through path traversal and anchor matching, and generates grounded answers using Google Gemini with an interactive local chat interface .

---

## Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites & Installation](#prerequisites--installation)
- [Environment Configuration](#environment-configuration)
- [How to Run](#how-to-run)
  - [1. Launch the Chat Interface (Lumen UI)](#1-launch-the-chat-interface-lumen-ui)
  - [2. Build the Knowledge Graph from Documents](#2-build-the-knowledge-graph-from-documents)
  - [3. Build Graph from Extracted PDF Corpus](#3-build-graph-from-extracted-pdf-corpus)
  - [4. Query Traversal via CLI](#4-query-traversal-via-cli)
- [Running Tests](#running-tests)
- [Architecture & Data Flow](#architecture--data-flow)

---

## Project Structure

```text
ContexPruning-LLM/
├── README.md                      # Project overview and run guide
├── AGENTS.md                      # Repository conventions and agent instructions
├── ECAP-RAG_PRD_v2.1_*.pdf        # Product requirements and specification documents
├── References/                    # Research references and background materials
├── Research Papers/               # Background academic papers
├── Synopsis/                      # Synopsis documentation
└── g_attention_rag/               # Primary application package
    ├── .env                       # Local secrets (API keys, models; git-ignored)
    ├── requirements.txt           # Python dependencies
    ├── main.py                    # CLI entry point for graph building, traversal, and plotting
    ├── build_all_pdfs.py          # Builds corpus knowledge graph from all extracted PDFs
    ├── graph_visualization.py     # Graph layout, styling, and PNG rendering
    ├── data/                      # Graph artifacts and raw source documents
    │   ├── knowledge_graph.pkl    # Serialized MultiDiGraph, claims, and traversal backbone
    │   ├── processed_graph/       # JSON exports and rendered PNG diagrams
    │   │   ├── graph_output.json  # Full readable knowledge graph state
    │   │   └── knowledge_graph.png# Visualized traversal graph
    │   └── raw/                   # Extracted and raw text corpora
    │       └── all_pdfs_text/     # Extracted text files per document/quarter
    ├── scripts/                   # Utility automation scripts
    │   └── extract_all_pdfs.ps1   # PowerShell PDF text extractor
    ├── src/                       # Core knowledge graph pipeline
    │   ├── graph_builder.py       # Public graph construction API
    │   ├── corpus_builder.py      # Multi-source alias resolution and graph merging
    │   ├── document_graph.py      # Single-document chunking and graph creation
    │   ├── claim_extraction.py    # spaCy dependency-based SVO claim extraction
    │   ├── claim_retrieval.py     # Claim deduplication and source passage binding
    │   ├── entity_aliases.py      # Entity mention resolution and alias tracking
    │   ├── graph_structure.py     # Nodes, context bridges, chunk nodes, and invariants
    │   ├── graph_traversal.py     # Retrieval entry point for path search
    │   ├── path_traversal.py      # Bounded BFS/path traversal over graph backbone
    │   ├── query_parser.py        # Entity & relation anchor extraction from questions
    │   ├── graph_constants.py     # Schema version and relation constants
    │   ├── graph_text.py          # Text canonicalization and paragraph slicing
    │   └── traversal_types.py     # Anchors, path records, and typed data structures
    ├── tests/                     # Unit test suite
    │   ├── test_ui.py             # UI server, Gemini mock, and endpoint tests
    │   ├── test_graph_builder.py  # Graph invariants, claims, and edge tests
    │   ├── test_graph_traversal.py# Path finding and claim retrieval tests
    │   └── test_query_parser.py   # Anchor extraction and parser tests
    └── ui/                        # Lumen chat interface
        ├── server.py              # Threaded HTTP server serving API and static assets
        ├── retrieval.py           # Adapts graph traversal for multi-turn chat
        ├── prompt.py              # System prompt and claim-grounded JSON context builder
        ├── gemini.py              # Gemini REST client (failover, retries, error mapping)
        ├── settings.py            # Environment loader and security limits
        └── static/                # Frontend web application
            ├── index.html         # Accessible dark-mode chat layout
            ├── styles.css         # Modern responsive styling
            └── app.js             # Form handling, local history, and error recovery
```

---

## Prerequisites & Installation

1. **Python 3.10+** (Python 3.10, 3.11, 3.12, 3.13, or 3.14).
2. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install requirements and the spaCy English model:

```powershell
cd g_attention_rag
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Environment Configuration

Create a `.env` file in `g_attention_rag/` (alongside `main.py`):

```dotenv
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

- **`GEMINI_API_KEY`**: Obtainable from [Google AI Studio](https://aistudio.google.com/).
- **`GEMINI_MODEL`** (Optional): Model name (defaults to `gemini-2.5-flash` or `gemini-1.5-flash`).

---

## How to Run

> **Note:** Always run commands from the `g_attention_rag/` directory.

### 1. Launch the Chat Interface (Lumen UI)

Start the local server:

```powershell
python -B -m ui.server
```

- Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.
- To specify a custom port:
  ```powershell
  python -B -m ui.server --port 8080
  ```
- **How it works**:
  1. Loads `data/knowledge_graph.pkl` on the first query.
  2. Parses the query to find entity/relation anchors.
  3. Traverses the knowledge graph to extract verified SVO claims and source passages.
  4. Passes grounded context to Google Gemini.
  5. Displays the stream-safe plain-text answer in the chat view with local conversation persistence.

### 2. Build the Knowledge Graph from Documents

To build or rebuild the knowledge graph from an input text file:

```powershell
python main.py path/to/document.txt
```

Options:
- `--layout {shell,hierarchical,spring,kamada}`: Visualization layout (default: `shell`).
- `--max-chunk-chars 1200`: Text chunk size limit.
- `--output-dir data`: Destination directory for serialized output.
- `--full-graph-plot`: Plot all edges in the generated visualization.

Output files saved:
- `data/knowledge_graph.pkl`: Complete serialized graph and claim state.
- `data/processed_graph/graph_output.json`: Full JSON representation of nodes, edges, claims, and chunks.
- `data/processed_graph/knowledge_graph.png`: Traversal backbone visualization image.

### 3. Build Graph from Extracted PDF Corpus

To process all text files extracted from financial/research reports:

```powershell
python build_all_pdfs.py --text-dir data/raw/all_pdfs_text --output-dir data
```

### 4. Query Traversal via CLI

You can also run query parsing and path retrieval directly from the terminal without opening the web browser:

```powershell
python main.py --retrieve --query "What are the financial results of Info Edge?"
```

---

## Running Tests

Run the complete automated unit test suite from `g_attention_rag/`:

```powershell
python -B -m unittest discover -s tests
```

To run individual test modules:

```powershell
# Test UI routes, validation, and Gemini adapter
python -B -m unittest discover -s tests -p test_ui.py -v

# Test Graph Builder logic and invariants
python -B -m unittest discover -s tests -p test_graph_builder.py -v

# Test Path Traversal and Claim Retrieval
python -B -m unittest discover -s tests -p test_graph_traversal.py -v

# Test Query Parser anchor extraction
python -B -m unittest discover -s tests -p test_query_parser.py -v
```

---

## Architecture & Data Flow

```text
[ Raw Documents / PDFs ]
          │
          ▼
   [ Text Chunker ]
          │
          ▼
[ spaCy Dependency Parsing ] ──► [ SVO Claim Extraction ]
          │                               │
          ▼                               ▼
[ Entity & Alias Resolution ] ──► [ MultiDiGraph Construction ]
                                          │
                                          ▼
                               [ knowledge_graph.pkl ]
                                          │
           ┌──────────────────────────────┴──────────────────────────────┐
           ▼                                                             ▼
 [ CLI Path Traversal ]                                        [ Lumen Web UI ]
           │                                                             │
  (Anchor matching &                                            (server.py / app.js)
   subgraph pruning)                                                     │
           │                                                             ▼
           ▼                                              [ Bounded Context Prompt ]
  [ Claim & Chunk Evidence ]                                             │
                                                                         ▼
                                                                [ Google Gemini LLM ]
                                                                         │
                                                                         ▼
                                                                [ Grounded Answer ]
```
